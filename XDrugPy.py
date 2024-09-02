"""
XDrugPy: protein hotspot analysis
https://tiny.cc/XDrugPy

CONTACT
    Pedro Sousa Lacerda <pslacerda@gmail.com>
    Marcelo Santos Castilho <castilho@ufba.br>

LICENSE
    CC BY-NC-SA 4.0 or Commercial License

REQUIREMENTS
    Incentive PyMOL 2.6+

"""

import platform
import subprocess
import sys
import os.path
from pymol import Qt


#
# INSTALL FPOCKET
#
QStandardPaths = Qt.QtCore.QStandardPaths
data_dir = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
system = platform.system().lower()
match system:
    case "windows":
        bin_fname = "fpocket.exe"
    case "linux" | "darwin":
        bin_fname = "fpocket"
fpocket_bin = f"{data_dir}/{bin_fname}"

if not os.path.exists(fpocket_bin):
    import os
    import stat
    from urllib.request import urlretrieve

    print(f'Installing Fpocket on "{fpocket_bin}"')
    fpocket_url = f"https://raw.githubusercontent.com/pslacerda/XDrugPy/master/bin/fpocket.{system}"
    urlretrieve(fpocket_url, fpocket_bin)
    os.chmod(fpocket_bin, stat.S_IEXEC)


#
# INSTALL OTHER REQUIREENTS
#

try:
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance_matrix, distance
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import pearsonr
    from matplotlib import pyplot as plt
    import seaborn as sb
    from strenum import StrEnum

except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "scipy",
            "jinja2",
            "matplotlib",
            "seaborn",
            "pandas",
            "openpyxl",
            "StrEnum",
        ],
    )


#
# CODE STARTS HERE
#

import tempfile
import os.path
import re
from fnmatch import fnmatch
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.spatial import distance_matrix, distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from pymol import cmd as pm, parsing
from matplotlib import pyplot as plt
import seaborn as sb
from strenum import StrEnum

__all__ = [
    "load_ftmap",
    "fo",
    "dc",
    "dce",
    "fp_sim",
    "ho",
    "res_sim",
    "hs_proj",
    "plot_dendrogram",
    "plot_heatmap",
]

matplotlib.use("Qt5Agg")

ONE_LETTER ={
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F',
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T',
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}

class Selection(str):
    pass


def _bool_func(value: str):
    if isinstance(value, str):
        if value.lower() in ["yes", "1", "true", "on"]:
            return True
        elif value.lower() in ["no", "0", "false", "off"]:
            return False
        else:
            raise Exception("Invalid boolean value: %s" % value)
    elif isinstance(value, bool):
        return value
    else:
        raise Exception(f"Unsuported boolean flag {value}")


def _add_completion(name, idx, sc):
    try:
        rec = pm.auto_arg[idx]
    except IndexError:
        rec = {}
        pm.auto_arg.append(rec)
    rec[name] = [sc, "var", ""]


def declare_command(name, function=None, _self=pm):
    if function is None:
        name, function = name.__name__, name

    if function.__code__.co_argcount != len(function.__annotations__):
        raise Exception("Messy annotations")
    from functools import wraps
    import inspect
    from pathlib import Path
    from enum import Enum
    import traceback

    spec = inspect.getfullargspec(function)

    kwargs_ = {}
    args_ = spec.args[:]

    defaults = list(spec.defaults or [])

    args2_ = args_[:]
    while args_ and defaults:
        kwargs_[args_.pop(-1)] = defaults.pop(-1)

    funcs = {}
    for idx, (var, func) in enumerate(spec.annotations.items()):
        funcs[var] = func

    @wraps(function)
    def inner(*args, **kwargs):
        frame = traceback.format_stack()[-2]
        caller = frame.split('"', maxsplit=2)[1]
        if caller.endswith("pymol/parser.py"):
            kwargs = {**kwargs_, **kwargs, **dict(zip(args2_, args))}
            kwargs.pop("_self", None)
            for arg in kwargs.copy():
                if funcs[arg] is _bool_func or issubclass(funcs[arg], bool):
                    funcs[arg] = _bool_func
                kwargs[arg] = funcs[arg](kwargs[arg])
            return function(**kwargs)
        else:
            return function(*args, **kwargs)

    name = function.__name__
    _self.keyword[name] = [inner, 0, 0, ",", parsing.STRICT]
    _self.kwhash.append(name)
    _self.help_sc.append(name)
    return inner


class FTMapSource(StrEnum):
    ATLAS = "atlas"
    FTMAP = "ftmap"
    EFTMAP = "eftmap"


def get_clusters():
    clusters = []
    eclusters = []

    for obj in pm.get_object_list():
        if obj.startswith(f"crosscluster."):
            _, _, s, _ = obj.split(".", maxsplit=4)
            coords = pm.get_coords(obj)
            clusters.append(
                SimpleNamespace(
                    source=FTMapSource.FTMAP,
                    selection=obj,
                    strength=int(s),
                    coords=coords,
                )
            )
        elif obj.startswith("consensus."):
            _, _, s = obj.split(".", maxsplit=3)
            coords = pm.get_coords(obj)
            clusters.append(
                SimpleNamespace(
                    source=FTMapSource.ATLAS,
                    selection=obj,
                    strength=int(s),
                    coords=coords,
                )
            )

        elif obj.startswith("clust."):
            source = FTMapSource.EFTMAP
            _, idx, s, probe_type = obj.split(".", maxsplit=4)
            eclusters.append(
                SimpleNamespace(
                    source=FTMapSource.EFTMAP,
                    selection=obj,
                    probe_type=probe_type,
                    strength=int(s),
                    idx=int(idx),
                )
            )
    return clusters, eclusters

def get_kozakov2015(group, clusters, max_length):
    k15 = []
    for length in range(max_length, 1, -1):
        for combination in combinations(clusters, length):
            cd = []
            cluster1 = combination[0]
            avg1 = np.average(cluster1.coords, axis=0)
            for cluster2 in combination:
                avg2 = np.average(cluster2.coords, axis=0)
                cd.append(distance.euclidean(avg1, avg2))

            coords = np.concatenate([c.coords for c in combination])
            max_coord = coords.max(axis=0)
            min_coord = coords.min(axis=0)

            selection = " or ".join(c.selection for c in combination)
            hs = SimpleNamespace(
                selection=selection,
                clusters=combination,
                kozakov_class=None,
                strength=sum(c.strength for c in combination),
                strength0=combination[0].strength,
                center_center=np.max(cd),
                max_dist=distance.euclidean(max_coord, min_coord),
                length=len(combination),
            )
            s0 = hs.clusters[0].strength
            sz = hs.clusters[-1].strength
            cd = hs.center_center
            md = hs.max_dist

            if s0 < 13 or md < 7 or sz <= 5:
                continue
            if s0 >= 16 and cd < 8 and md >= 10:
                hs.kozakov_class = "D"
            if s0 >= 16 and cd < 8 and 7 <= md < 10:
                hs.kozakov_class = "DS"
            if 13 <= s0 < 16 and cd < 8 and md >= 10:
                hs.kozakov_class = "B"
            if 13 <= s0 < 16 and cd < 8 and 7 <= md < 10:
                hs.kozakov_class = "BS"

            if hs.kozakov_class:
                k15.append(hs)

    k15 = sorted(k15, key=lambda hs: (-hs.strength0, -hs.strength))
    k15 = sorted(k15, key=lambda hs: ["D", "DS", "B", "BS"].index(hs.kozakov_class))
    k15 = list(k15)

    for idx, hs in enumerate(k15):
        new_name = f"{group}.{hs.kozakov_class}_{idx:02}"
        pm.create(new_name, hs.selection)
        pm.group(group, new_name)
        hs.selection = new_name

        pm.set_property("Type", "Kozakov2015", new_name)
        pm.set_property("Group", group, new_name)
        pm.set_property("Class", hs.kozakov_class, new_name)
        pm.set_property("S", hs.strength, new_name)
        pm.set_property("S0", hs.strength0, new_name)
        pm.set_property("CD", round(hs.center_center, 2), new_name)
        pm.set_property("MD", round(hs.max_dist, 2), new_name)
        pm.set_property("Length", hs.length, new_name)

    return k15


def get_fpocket(group, protein):
    pockets = []
    with tempfile.TemporaryDirectory() as tempdir:
        protein_pdb = f"{tempdir}/{group}.pdb"
        pm.save(protein_pdb, selection=protein)
        subprocess.check_call(
            [fpocket_bin, "-f", protein_pdb],
            env={"TMPDIR": tempdir},
        )
        header_re = re.compile(r"^HEADER\s+\d+\s+-(.*):(.*)$")
        path = Path(tempdir)
        for pocket_pdb in path.glob(f"{group}_out/pockets/pocket*_atm.pdb"):
            idx = (
                pocket_pdb.name
                .replace("pocket", "")
                .replace("_atm.pdb", "")
            )
            idx = int(idx)
            pocket = SimpleNamespace(selection=f"{group}.fpocket_{idx:02}")
            pm.delete(pocket.selection)
            pm.load(pocket_pdb, pocket.selection)
            pm.set_property("Type", "Fpocket", pocket.selection)
            pm.set_property("Group", group, pocket.selection)
            for line in pm.get_property("pdb_header", pocket.selection).split("\n"):
                if match := header_re.match(line):
                    prop = match.group(1).strip()
                    value = float(match.group(2))
                    setattr(pocket, prop, value)
                    pm.set_property(prop, value, pocket.selection)
            pockets.append(pocket)

    return pockets


def process_clusters(group, clusters):
    for idx, cs in enumerate(clusters):
        new_name = f"{group}.CS_{idx:03}_{cs.strength:03}"
        pm.create(new_name, cs.selection)
        pm.group(group, new_name)

        pm.set_property("Type", "CS", new_name)
        pm.set_property("Group", group, new_name)
        pm.set_property("S", cs.strength, new_name)

    pm.delete("consensus.*")
    pm.delete("crosscluster.*")


def process_eclusters(group, eclusters):
    for acs in eclusters:
        new_name = f"{group}.ACS_{acs.probe_type}_{acs.idx:02}_{acs.strength:02}"
        pm.create(new_name, acs.selection)
        pm.group(group, new_name)

        coords = pm.get_coords(new_name)
        md = distance_matrix(coords, coords).max()

        pm.set_property("Type", "ACS", new_name)
        pm.set_property("Group", group, new_name)
        pm.set_property("ProbeType", acs.probe_type, new_name)
        pm.set_property("S", acs.strength, new_name)
        pm.set_property("MD", round(md, 2), new_name)

    pm.delete("clust.*")


def get_egbert2019(group, fpo_list, clusters):
    e19 = []
    idx = 0
    for i, pocket in enumerate(fpo_list):
        sel = f"byobject ({group}.CS_* within 4 of {pocket.selection})"
        objs = pm.get_object_list(sel)
        if len(objs) > 3 and sum([pm.get_property("S", o) >= 16 for o in objs]) > 2:
            new_name = f"{group}.C_{idx:02}"
            pm.create(new_name, sel)
            pm.group(group, new_name)

            s_list = [pm.get_property("S", o) for o in objs]
            pm.set_property("Type", "Egbert2019", new_name)
            pm.set_property("Group", group, new_name)
            pm.set_property("Fpocket", pocket.selection, new_name)
            pm.set_property("S", sum(s_list), new_name)
            pm.set_property("S0", s_list[0])
            pm.set_property("S1", s_list[1])
            pm.set_property("Length", len(objs), new_name)
            e19.append(SimpleNamespace(selection=new_name))
            idx += 1
    return e19


@declare_command
def load_ftmap(
    filename: Path,
    group: str = "",
    kozakov2015_max_length: int = 8,
    fpocket: bool = True
):
    """
    Load a FTMap PDB file and classify hotspot ensembles in accordance to
    Kozakov et al. (2015).
    https://doi.org/10.1021/acs.jmedchem.5b00586

    OPTIONS
        filename    mapping PDB file.
        group       optional group name.

    EXAMPLES
        load_ftmap ace_example.pdb
        load_ftmap ace_example.pdb, group=MyProtein
    """
    if not group:
        group = os.path.splitext(os.path.basename(filename))[0]
    group = group.replace(".", "_")

    pm.delete(f"%{group}")
    pm.load(filename, quiet=1)

    if objs := pm.get_object_list("*_protein"):
        assert len(objs) == 1
        protein = objs[0]
    elif pm.get_object_list("protein"):
        protein = "protein"
    pm.set_name(protein, f"{group}.protein")
    pm.group(group, f"{group}.protein")

    clusters, eclusters = get_clusters()
    k15_list = get_kozakov2015(group, clusters, kozakov2015_max_length)
    if fpocket:
        fpo_list = get_fpocket(group, f"{group}.protein")
    else:
        fpo_list = None
    process_clusters(group, clusters)
    process_eclusters(group, eclusters)
    if fpocket:
        e19_list = get_egbert2019(group, fpo_list, clusters)
    else:
        e19_list = None

    pm.hide("everything", f"{group}.*")

    pm.show("cartoon", f"{group}.protein")
    pm.show("mesh", f"{group}.D* or {group}.B*")
    pm.show("spheres", f"{group}.ACS_*")
    pm.show("spheres", f"{group}.fpocket_*")
    pm.set("sphere_scale", 0.25, f"{group}.ACS_*")
    pm.set("sphere_scale", 0.2, f"{group}.fpocket_*")

    pm.color("red", f"{group}.D*")
    pm.color("salmon", f"{group}.B*")
    pm.color("red", f"{group}.ACS_acceptor_*")
    pm.color("blue", f"{group}.ACS_donor_*")
    pm.color("green", f"{group}.ACS_halogen_*")
    pm.color("orange", f"{group}.ACS_aromatic_*")
    pm.color("yellow", f"{group}.ACS_apolar_*")
    pm.color("white", f"{group}.fpocket_*")

    pm.disable(f"{group}.CS_*")
    pm.disable(f"{group}.fpocket_*")
    pm.show("line", f"{group}.CS*")

    pm.set("mesh_mode", 1)
    pm.orient("all")

    return SimpleNamespace(
        clusters=clusters,
        eclusters=eclusters,
        kozakov2015=k15_list,
        egbert2019=e19_list,
        fpocket=fpo_list,
    )


@pm.extend
def count_molecules(sel):
    """
    Returns the number of distinct molecules in a given selection.
    """

    sel_copy = "__selcopy"
    pm.select(sel_copy, sel)
    num_objs = 0
    atoms_in_sel = pm.count_atoms(sel_copy)
    while atoms_in_sel > 0:
        num_objs += 1
        pm.select(sel_copy, "%s and not (bm. first %s)" % (sel_copy, sel_copy))
        atoms_in_sel = pm.count_atoms(sel_copy)
    return num_objs


@declare_command
def fo(
    sel1: Selection,
    sel2: Selection,
    radius: float = 2,
    verbose: bool = True,
):
    """
    Compute the fractional overlap of sel1 respective to sel2.
        FO = Nc/Nt

    Nc is the number of atoms of sel1 in contact with sel2. Nt is the number of atoms
    of sel1. Hydrogen atoms are ignored.

    OPTIONS
        sel1    ligand object.
        sel2    hotspot object.
        radius  the radius so sel1 and sel2 are in contact (default: 2).

    EXAMPLE
        fo REF_LIG, ftmap1234.D_003_*_*
    """
    atoms1 = pm.get_coords(f"({sel1}) and not elem H")
    atoms2 = pm.get_coords(f"({sel2}) and not elem H")
    dist = distance_matrix(atoms1, atoms2) - radius <= 0
    num_contacts = np.sum(np.any(dist, axis=1))
    total_atoms = len(atoms1)
    fo_ = num_contacts / total_atoms
    if verbose:
        print(f"FO: {fo_:.2f}")
    return fo_


@declare_command
def dc(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    verbose: bool = True,
):
    """
    Compute the Density Correlation according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are the selections representing the molecules or hotspots. The
    threshold distance can be changed with radius.

    OPTIONS
        sel1    first object
        sel2    second object
        radius  the radius so two atoms are in contact (default: 1.25)
        verbose define verbosity

    EXAMPLES
        dc REF_LIG, ftmap1234.D_003_*_*
        dc ftmap1234.D.003, REF_LIG, radius=1.5

    """
    xyz1 = pm.get_coords(f"({sel1}) and not elem H")
    xyz2 = pm.get_coords(f"({sel2}) and not elem H")

    dc_ = (distance_matrix(xyz1, xyz2) < radius).sum()
    if verbose:
        print(f"DC: {dc_:.2f}")
    return dc_


@declare_command
def dce(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    verbose: bool = True,
):
    """
    Compute the Density Correlation Efficiency according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are respectively the molecule and hotspot. The threshold
    distance can be changed with radius.

    OPTIONS
        sel1    ligand object
        sel2    hotspot object
        radius  the radius so two atoms are in contact (default: 1.25)
        verbose define verbosity

    EXAMPLE
        dce REF_LIG, ftmap1234.D_003_*_*
    """
    dc_ = dc(sel1, sel2, radius, verbose=False)
    dce_ = dc_ / pm.count_atoms(f"({sel1}) and not elem H")
    if verbose:
        print(f"DCE: {dce_:.2f}")
    return dce_


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"


@declare_command
def fp_sim(
    hotspots: Selection,
    site: str = "*",
    radius: int = 4,
    plot_fingerprints: bool = True,
    nbins: int = 5,
    plot_dendrogram: bool = False,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    verbose: bool = True,
):
    """
    Compute the similarity between the residue contact fingerprint of two
    hotspots.

    OPTIONS:
        hotspots          hotspot expressions
        site              selection to focus based on first protein
        radius            radius to compute the contacts (default: 4)
        plot_fingerprints plot the fingerprints (default: True)
        nbins             number of residue labels (default: 5)
        plot_dendrogram   plot the dendrogram (default: False)
        linkage_method    linkage method (default: single)
        verbose           define verbosity

    EXAMPLES
        fs_sim 8DSU.D_01* 6XHM.D_01*
        fs_sim 8DSU.CS_* 6XHM.CS_*, site=resi 8-101, nbins=10
    """

    expanded_hss = []
    all_groups = [g.split(".", maxsplit=1)[0] for g in pm.get_object_list("*.protein")]
    for expr in hotspots.split():
        expr_g, expr_part = expr.split(".", maxsplit=1)
        for g in all_groups:
            if fnmatch(g, expr_g):
                expanded_hss.append("%s.%s" % (g, expr_part))

    hotspots = expanded_hss
    groups = [hs.split(".", maxsplit=1)[0] for hs in hotspots]
    proteins = [f"{g}.protein" for g in groups]

    p0 = proteins[0]
    site_index = set()
    pm.iterate(
        f"({p0}) and name CA and ({site})",
        "site_index.add(index)",
        space={"site_index": site_index}
    )

    def get_resi_map(p1, p2):
        try:
            aln_obj = pm.get_unused_name()
            pm.cealign(p1, p2, transform=0, object=aln_obj)
            raw = pm.get_raw_alignment(aln_obj)
        finally:
            pm.delete(aln_obj)
        resis2 = {}
        pm.iterate(
            p2,
            "resis2[index] = (model, index, resn, resi, chain)",
            space={"resis2": resis2},
        )
        resis_map = {}
        for (model1, idx1), (model2, idx2) in raw:
            if idx1 not in site_index:
                continue
            resis_map[idx1] = resis2[idx2]
        return resis_map
    
    def calc_fp(hs, resi_map):
        fpt = []
        labels = []
        for index in resi_map:
            model, mapped_index, resn, resi, chain = resi_map[index]
            cnt = count_molecules(
                f"({hs}) within {radius} from (byres %{model} and index {mapped_index})"
            )
            fpt.append(cnt)
            resn = ONE_LETTER.get(resn, 'X')
            labels.append(f"{resn}{resi}{chain}")
        return fpt, labels

    def plot_fp(fp, lbl, hs, ax):
        ax.set_ylabel(hs)
        ax.bar(np.arange(len(fp)), fp)
        ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
        ax.set_xticks(np.arange(len(fp)), labels=lbl, rotation=45)
        ax.locator_params(axis="x", tight=True, nbins=nbins)
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment("right")

    plt.close()
    if plot_fingerprints and plot_dendrogram:
        fig, axd = plt.subplot_mosaic(list(
            zip(range(len(proteins)), ["DENDRO"] * len(proteins))
        ))
    elif plot_fingerprints and not plot_dendrogram:
        fig, axd = plt.subplot_mosaic(list(zip(range(len(proteins)))))
    elif not plot_fingerprints and plot_dendrogram:
        fig, axd = plt.subplot_mosaic([["DENDRO"]])
        
    fp_list = []
    for i, (p, hs) in enumerate(zip(proteins, hotspots)):
        fp, lbl = calc_fp(hs, get_resi_map(p0, p))
        fp_list.append(fp)
        if plot_fingerprints:
            plot_fp(fp, lbl, hs, axd[i])
    
    fp0 = fp_list[0]
    if not all([len(fp0) == len(fp) for fp in fp_list]):
        raise ValueError(
            "All fingerprints must have the same length. "
            "Do you have incomplete structures?"
        )
    
    if verbose or plot_dendrogram:
        cor_list = []
        for idx1, (fp1, hs1) in enumerate(zip(fp_list, hotspots)):
            for idx2, (fp2, hs2) in enumerate(zip(fp_list, hotspots)):
                if idx1 >= idx2:
                    continue
                cor = pearsonr(fp1, fp2).statistic
                if np.isnan(cor):
                    cor = 0
                cor_list.append(cor)
                if verbose:
                    print(f"Pearson correlation: {hs1} / {hs2}: {cor:.2f}")
            
        if plot_dendrogram:
            dendrogram(
                linkage([1 - c for c in cor_list], method=linkage_method),
                labels=hotspots,
                ax=axd["DENDRO"],
                leaf_rotation=45,
                color_threshold=0,
            )
    
    plt.tight_layout()
    plt.show()
    return fp_list


@declare_command
def ho(
    hs1: Selection,
    hs2: Selection,
    radius: float = 2.5,
    verbose: bool = True,
):
    """
    Compute the Hotspot Overlap (HO) metric. HO is defined as the number of
    atoms in hs1 in contact with hs2 plus the number of atoms in hs2 in
    contact with hs1 divided by the total number of atoms in both hotspots.

    OPTIONS
        hs1     an hotspot object
        hs2     another hotspot object
        radius  the distance to consider two atoms in contact (default: 2.5)
        verbose define verbosity
    """
    atoms1 = pm.get_coords(f"({hs1}) and not elem H")
    atoms2 = pm.get_coords(f"({hs2}) and not elem H")
    dist = distance_matrix(atoms1, atoms2) - radius <= 0
    num_contacts1 = np.sum(np.any(dist, axis=1))
    num_contacts2 = np.sum(np.any(dist, axis=0))
    ho = (num_contacts1 + num_contacts2) / (len(atoms1) + len(atoms2))
    if verbose:
        print(f"HO: {ho:.2f}")
    return ho


class ResidueSimilarityMethod(StrEnum):
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@declare_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: int = 2,
    method: ResidueSimilarityMethod = ResidueSimilarityMethod.JACCARD,
    verbose: bool = True,
):
    """
    Compute hotspots similarity by the Jaccard or overlap coefficient of nearby
    residues.

    OPTIONS
        hs1     hotspot 1
        hs2     hotspot 2
        radius  distance to consider residues near hotspots (default: 2)
        method  jaccard or overlap (default: jaccard)
        verbose define verbosity

    EXAMPLES
        res_sim 8DSU.D_001*, 6XHM.D_001*
        res_sim 8DSU.CS_*, 6XHM.CS_*
    """

    group1 = hs1.split(".", maxsplit=1)[0]
    group2 = hs2.split(".", maxsplit=1)[0]

    sel1 = f"{group1}.protein within {radius} from ({hs1})"
    sel2 = f"{group2}.protein within {radius} from ({hs2})"

    resis1 = set()
    pm.iterate(sel1, "resis1.add((chain, resi))", space={"resis1": resis1})

    if group1 == group2:
        resis2 = set()
        pm.iterate(sel2, "resis2.add((chain, resi))", space={"resis2": resis2})
    else:
        try:
            aln_obj = pm.get_unused_name()
            pm.cealign(
                f"{group1}.protein", f"{group2}.protein", transform=0, object=aln_obj
            )
            raw = pm.get_raw_alignment(aln_obj)

            resis = {}
            pm.iterate(
                aln_obj, "resis[model, index] = (chain, resi)", space={"resis": resis}
            )

            site2 = [(a.chain, a.resi) for a in pm.get_model(sel2).atom]
            resis2 = set()
            for idx1, idx2 in raw:
                if resis[idx1] in site2:
                    resis2.add(resis[idx2])
        finally:
            pm.delete(aln_obj)

    try:
        match method:
            case ResidueSimilarityMethod.JACCARD:
                ret = len(resis1.intersection(resis2)) / len(resis1.union(resis2))
            case ResidueSimilarityMethod.OVERLAP:
                ret = len(resis1.intersection(resis2)) / min(len(resis1), len(resis2))
    except ZeroDivisionError:
        print("Your selection yields zero atoms.")
        return 0.0

    if verbose:
        print(f"{method} similarity: {ret:.2}")
    return ret


class HeatmapFunction(StrEnum):
    HO = "ho"
    RESIDUE_JACCARD = "residue_jaccard"
    RESIDUE_OVERLAP = "residue_overlap"


@declare_command
def plot_heatmap(
    objs: Selection,
    method: HeatmapFunction = HeatmapFunction.HO,
    radius: float = 2.0,
    annotate: bool = False,
):
    """
    Compute the similarity between matching objects using a similarity function.

    OPTIONS
        objs        space separated list of object expressions
        method      ho, residue_jaccard, or residue_overlap (default: ho) 
        radius      the radius to consider atoms in contact (default: 2.0)
        annotate    fill the cells with values

    EXAMPLES
        cross_measure *.D_000_*_*, function=residue_jaccard
        cross_measure *.D_*
        cross_measure *.D_000_*_* *.DS_*
    """
    objs = objs.split(" ")

    obj1s = []
    for obj in pm.get_object_list():
        for obj_sub in objs:
            if not fnmatch(obj, obj_sub):
                continue
            obj1s.append(obj)

    def sort(obj):
        klass = pm.get_property("Class", obj)
        return str(klass), obj

    obj1s = list(sorted(obj1s, key=sort))

    if len(obj1s) == 0:
        raise ValueError("No objects found")

    mat = []
    for idx1, obj1 in enumerate(obj1s):
        mat.append([])
        for idx2, obj2 in enumerate(obj1s):
            if idx1 == idx2:
                ret = 1
            elif idx2 > idx1:
                ret = np.nan
            else:
                match method:
                    case HeatmapFunction.HO:
                        ret = ho(obj1, obj2, radius=radius, verbose=False)
                    case HeatmapFunction.RESIDUE_JACCARD:
                        ret = res_sim(obj1, obj2, radius=radius, method="jaccard", verbose=False)
                    case HeatmapFunction.RESIDUE_OVERLAP:
                        ret = res_sim(obj1, obj2, radius=radius, method="overlap", verbose=False)
            mat[-1].append(round(ret, 2))

    plt.close()
    fig, ax = plt.subplots(1)
    sb.heatmap(
        mat,
        yticklabels=obj1s,
        xticklabels=obj1s,
        cmap="viridis",
        annot=annotate,
        ax=ax,
    )
    plt.xticks(rotation=45)
    for label in ax.xaxis.get_majorticklabels():
        label.set_horizontalalignment("right")

    plt.tight_layout()
    plt.show()
    return mat


class PrioritizationType(StrEnum):
    RESIDUE = "residue"
    ATOM = "atom"


@declare_command
def hs_proj(
    sel: Selection,
    protein: Selection = "",
    radius: int = 4,
    type: PrioritizationType = PrioritizationType.RESIDUE,
    palette: str = "rainbow",
):
    """
    Colour atoms by proximity with FTMap probes.

    OPTIONS:
        sel         probes selection.
        protein     object which will be coloured.
        max_dist    maximum distance in Angstroms (default: 4).
        type        residue or type (default: residue).
        palette     spectrum colour palette (default: rainbow).
    """

    if not protein:
        group = sel.split(".", maxsplit=1)[0]
        protein = f"{group}.protein"

    pm.alter(protein, "q=0")
    for prot_atom in pm.get_model(f"({protein}) within {radius} of ({sel})").atom:
        match type:
            case PrioritizationType.RESIDUE:
                prot_atom_sel = f"byres index {prot_atom.index}"
            case PrioritizationType.ATOM:
                prot_atom_sel = f"index {prot_atom.index}"
        count = count_molecules(f"({sel}) within {radius} of ({prot_atom_sel})")
        pm.alter(prot_atom_sel, f"q={count}")

    pm.hide("everything", protein)
    match type:
        case PrioritizationType.RESIDUE:
            pm.show("surface", protein)
        case PrioritizationType.ATOM:
            pm.show("cartoon", protein)
            pm.show("sticks", "byres q>0")
    pm.spectrum("q", palette=palette, selection=protein)


@declare_command
def plot_dendrogram(
    exprs: Selection,
    com_weight: float = 1,
    residue_radius: int = 4,
    residue_weight: float = 1,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0,
):
    """
    Compute the similarity dendrogram of hotspots.
    OPTIONS
        exprs           space separated list of object expressions
        com_weight      center-of-mass (x, y, z) weight
        residue_radius  maximum distance for residue_similarity (default: 4)
        residue_weight  residue similarity weight (default: 5)
        method          linkage method: single, complete, average, or
                        (default: single)
    EXAMPLES
        plot_similarity *.D_* *.DS_*, method=complete
    """

    def _get_property_vector(hs_type, obj):
        x, y, z = np.mean(pm.get_coords(obj), axis=0)

        if hs_type == "Kozakov2015":
            S = pm.get_property("S", obj)
            S0 = pm.get_property("S0", obj)
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            return np.array([S, S0, CD, MD, x, y, z])
        elif hs_type == "CS":
            S = pm.get_property("S", obj)
            return np.array([S, x, y, z])
        elif hs_type == "ACS":
            S = pm.get_property("S", obj)
            MD = pm.get_property("MD", obj)
            return np.array([S, MD, x, y, z])

    def _euclidean_like(hs_type, p1, p2, j):
        if hs_type == "Kozakov2015":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + (p1[2] - p2[2]) ** 2
                + (p1[3] - p2[3]) ** 2
                + com_weight * (p1[4] - p2[4]) ** 2
                + com_weight * (p1[5] - p2[5]) ** 2
                + com_weight * (p1[6] - p2[6]) ** 2
                + residue_weight * (1 - j) ** 2
            )
        elif hs_type == "CS":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + com_weight * (p1[1] - p2[1]) ** 2
                + com_weight * (p1[2] - p2[2]) ** 2
                + com_weight * (p1[3] - p2[3]) ** 2
                + residue_weight * (1 - j) ** 2
            )
        elif hs_type == "ACS":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + com_weight * (p1[2] - p2[2]) ** 2
                + com_weight * (p1[3] - p2[3]) ** 2
                + com_weight * (p1[4] - p2[4]) ** 2
                + residue_weight * (1 - j) ** 2
            )

    object_list = []
    for expr in exprs.split(" "):
        for idx, obj in enumerate(pm.get_object_list()):
            if fnmatch(obj.lower(), expr.lower()):
                object_list.append(obj)
    assert len(set(pm.get_property("Type", o) for o in object_list)) == 1

    hs_type = pm.get_property("Type", object_list[0])
    if hs_type == "Kozakov2015":
        n_props = 4
    elif hs_type == "CS":
        n_props = 1
    elif hs_type == "ACS":
        n_props = 2

    labels = []
    p = np.zeros((len(object_list), n_props + 3))

    for idx, obj in enumerate(object_list):
        p[idx, :] = _get_property_vector(hs_type, obj)
        labels.append(obj)

    for col in range(n_props + 3):
        if max(p[:, col]) - min(p[:, col]) == 0:
            p[:, col] = 0
        else:
            p[:, col] = (p[:, col] - min(p[:, col])) / (max(p[:, col]) - min(p[:, col]))

    X = []
    for idx1, obj1 in enumerate(object_list):
        for idx2, obj2 in enumerate(object_list):
            if idx1 >= idx2:
                continue

            p1 = p[idx1, :]
            p2 = p[idx2, :]
            if residue_weight != 0:
                j = res_sim(obj1, obj2, radius=residue_radius, verbose=False)
            else:
                j = 0
            d = _euclidean_like(hs_type, p1, p2, j)
            X.append(d)

    plt.close()
    dendrogram(
        linkage(X, method=linkage_method),
        labels=labels,
        color_threshold=color_threshold,
        orientation="right",
    )
    plt.axvline(x=color_threshold, c="grey", lw=1, linestyle="dashed")
    plt.tight_layout()
    plt.show()
    return X, labels


#
# GRAPHICAL USER INTERFACE
#

from pymol import Qt

QWidget = Qt.QtWidgets.QWidget
QFileDialog = Qt.QtWidgets.QFileDialog
QFormLayout = Qt.QtWidgets.QFormLayout
QPushButton = Qt.QtWidgets.QPushButton
QSpinBox = Qt.QtWidgets.QSpinBox
QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
QLineEdit = Qt.QtWidgets.QLineEdit
QCheckBox = Qt.QtWidgets.QCheckBox
QVBoxLayout = Qt.QtWidgets.QVBoxLayout
QHBoxLayout = Qt.QtWidgets.QHBoxLayout
QDialog = Qt.QtWidgets.QDialog
QComboBox = Qt.QtWidgets.QComboBox
QTabWidget = Qt.QtWidgets.QTabWidget
QLabel = Qt.QtWidgets.QLabel
QTableWidget = Qt.QtWidgets.QTableWidget
QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
QGroupBox = Qt.QtWidgets.QGroupBox
QHeaderView = Qt.QtWidgets.QHeaderView

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon


class LoadWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Group", "Filename"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        addRemoveLayout = QHBoxLayout()
        layout.addLayout(addRemoveLayout)

        pickFileButton = QPushButton("Add")
        pickFileButton.clicked.connect(self.pickFile)
        addRemoveLayout.addWidget(pickFileButton)

        removeButton = QPushButton("Remove")
        removeButton.clicked.connect(self.removeRow)
        addRemoveLayout.addWidget(removeButton)

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.load)
        addRemoveLayout.addWidget(loadButton)

    def pickFile(self):
        fileDIalog = QFileDialog()
        fileDIalog.setFileMode(QFileDialog.ExistingFiles)
        fileDIalog.setNameFilter("FTMap PDB (*.pdb)")
        fileDIalog.setViewMode(QFileDialog.Detail)

        if fileDIalog.exec_():
            for filename in fileDIalog.selectedFiles():
                basename = os.path.splitext(os.path.basename(filename))
                group = basename[0]
                self.appendRow(filename, group)

    def appendRow(self, filename, group):
        groupItem = QTableWidgetItem(group)
        filenameItem = QTableWidgetItem(filename)

        filenameItem.setFlags(filenameItem.flags() & ~QtCore.Qt.ItemIsEditable)

        self.table.insertRow(self.table.rowCount())
        self.table.setItem(self.table.rowCount() - 1, 0, groupItem)
        self.table.setItem(self.table.rowCount() - 1, 1, filenameItem)

    def removeRow(self):
        self.table.removeRow(self.table.currentRow())

    def clearRows(self):
        self.table.setRowCount(0)

    def load(self):
        try:
            for row in range(self.table.rowCount()):
                group = self.table.item(row, 0).text()
                filename = self.table.item(row, 1).text()
                try:
                    load_ftmap(
                        filename,
                        group=group,
                    )
                except Exception:
                    try:
                        load_ftmap(
                            filename,
                            group=group,
                        )
                    except Exception:
                        if not os.path.exists(filename):
                            raise ValueError(f"File does not exist: '{filename}'")
                        else:
                            raise Exception(f"Failed to load file: '{filename}'")
        finally:
            self.clearRows()

class SortableItem(QTableWidgetItem):
    def __init__(self, obj):
        super().__init__(str(obj))
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()

class TableWidget(QWidget):

    class TableWidgetImpl(QTableWidget):
        def __init__(self, props):
            super().__init__()
            self.setSelectionBehavior(QTableWidget.SelectRows)
            self.setSelectionMode(QTableWidget.SingleSelection)
            self.setColumnCount(len(props) + 1)
            self.setHorizontalHeaderLabels(["Object"] + props)
            header = self.horizontalHeader()
            for idx in range(len(props) + 1):
                header.setSectionResizeMode(
                    idx, QHeaderView.ResizeMode.ResizeToContents
                )

            @self.itemClicked.connect
            def itemClicked(item):
                obj = self.item(item.row(), 0).text()
                pm.select(obj)
                pm.enable("sele")

        def hideEvent(self, evt):
            self.clearSelection()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        tab = QTabWidget()
        layout.addWidget(tab)

        self.hotspotsMap = {
            "Kozakov2015": ["Class", "S", "S0", "CD", "MD", "Length"],
            "CS": ["S"],
            "ACS": ["Class", "S", "MD"],
            "Egbert2019": ["Fpocket", "S", "S0", "S1", "Length"],
            "Fpocket": ["Pocket Score", "Drug Score"],
        }
        self.tables = {}
        for key, props in self.hotspotsMap.items():
            table = self.TableWidgetImpl(props)
            self.tables[key] = table
            tab.addTab(table, key)

        exportButton = QPushButton(QIcon("save"), "Export Tables")
        exportButton.clicked.connect(self.export)
        layout.addWidget(exportButton)

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)

    def refresh(self):
        for key, props in self.hotspotsMap.items():
            self.tables[key].setSortingEnabled(False)

            # remove old rows
            while self.tables[key].rowCount() > 0:
                self.tables[key].removeRow(0)

            # append new rows
            for obj in pm.get_object_list():
                obj_type = pm.get_property("Type", obj)
                if obj_type == key:
                    self.appendRow(key, obj)

            self.tables[key].setSortingEnabled(True)

    def appendRow(self, key, obj):
        self.tables[key].insertRow(self.tables[key].rowCount())
        line = self.tables[key].rowCount() - 1

        self.tables[key].setItem(line, 0, SortableItem(obj))

        for idx, prop in enumerate(self.hotspotsMap[key]):
            prop_value = pm.get_property(prop, obj)
            self.tables[key].setItem(line, idx + 1, SortableItem(prop_value))

    def export(self):
        fileDialog = QFileDialog()
        fileDialog.setNameFilter("Excel file (*.xlsx)")
        fileDialog.setViewMode(QFileDialog.Detail)
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDefaultSuffix(".xlsx")

        if fileDialog.exec_():
            filename = fileDialog.selectedFiles()[0]
            ext = os.path.splitext(filename)[1]
            with pd.ExcelWriter(filename) as xlsx_writer:
                for key, props in self.hotspotsMap.items():
                    data = {"Object": [], **{p: [] for p in props}}
                    for header in data:
                        column = list(data.keys()).index(header)
                        for line in range(self.tables[key].rowCount()):
                            item = self.tables[key].item(line, column)
                            data[header].append(self.parse_item(item))
                    df = pd.DataFrame(data)
                    df.to_excel(xlsx_writer, sheet_name=key, index=False)

    @staticmethod
    def parse_item(item):
        try:
            item = int(item.text())
        except ValueError:
            try:
                item = float(item.text())
            except ValueError:
                item = item.text()
        return item

class SimilarityWidget(QWidget):

    def __init__(self):
        super().__init__()

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        groupBox = QGroupBox("General")
        mainLayout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotExpressionLine = QLineEdit()
        boxLayout.addRow("Hotspots:", self.hotspotExpressionLine)

        layout = QHBoxLayout()
        mainLayout.addLayout(layout)

        groupBox = QGroupBox("Heatmap")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in HeatmapFunction])
        boxLayout.addRow("Function:", self.functionCombo)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.annotateCheck = QCheckBox()
        self.annotateCheck.setChecked(True)
        boxLayout.addRow("Annotate:", self.annotateCheck)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_heatmap)
        boxLayout.addWidget(plotButton)

        groupBox = QGroupBox("Dendrogram")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.comWeightSpin = QDoubleSpinBox()
        self.comWeightSpin.setValue(1.0)
        self.comWeightSpin.setSingleStep(0.1)
        self.comWeightSpin.setDecimals(1)
        self.comWeightSpin.setMinimum(0)
        self.comWeightSpin.setMaximum(20)
        boxLayout.addRow("COM weight:", self.comWeightSpin)

        self.residueRadiusSpin = QSpinBox()
        self.residueRadiusSpin.setValue(4)
        self.residueRadiusSpin.setMinimum(3)
        self.residueRadiusSpin.setMaximum(5)
        boxLayout.addRow("Residue radius:", self.residueRadiusSpin)

        self.residueWeightSpin = QDoubleSpinBox()
        self.residueWeightSpin.setValue(1.0)
        self.residueWeightSpin.setSingleStep(0.1)
        self.residueWeightSpin.setDecimals(1)
        self.residueWeightSpin.setMinimum(0)
        self.residueWeightSpin.setMaximum(20)
        boxLayout.addRow("Residue weight:", self.residueWeightSpin)

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems([e.value for e in LinkageMethod])
        boxLayout.addRow("Linkage:", self.linkageMethodCombo)

        self.colorThresholdSpin = QDoubleSpinBox()
        self.colorThresholdSpin.setValue(0.0)
        self.colorThresholdSpin.setSingleStep(0.1)
        self.colorThresholdSpin.setDecimals(1)
        self.colorThresholdSpin.setMinimum(0)
        boxLayout.addRow("Color threshold:", self.colorThresholdSpin)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_dendrogram)
        boxLayout.addWidget(plotButton)

    def plot_heatmap(self):
        expression = self.hotspotExpressionLine.text()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()

        plot_heatmap(expression, function, radius, annotate)

    def plot_dendrogram(self):
        expression = self.hotspotExpressionLine.text()
        com_weight = self.comWeightSpin.value()
        residue_radius = self.residueRadiusSpin.value()
        residue_weight = self.residueWeightSpin.value()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()

        plot_dendrogram(
            expression,
            com_weight,
            residue_radius,
            residue_weight,
            linkage_method,
            color_threshold,
        )

class CountWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("Residue projection")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotsExpressionLine = QLineEdit()
        boxLayout.addRow("Hotspots:", self.hotspotsExpressionLine)

        self.proteinExpressionLine = QLineEdit()
        boxLayout.addRow("Protein:", self.proteinExpressionLine)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(3)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.typeCombo = QComboBox()
        self.typeCombo.addItems([e.value for e in PrioritizationType])
        boxLayout.addRow("Type:", self.typeCombo)

        self.paletteLine = QLineEdit("rainbow")
        boxLayout.addRow("Palette:", self.paletteLine)

        drawButton = QPushButton("Draw")
        drawButton.clicked.connect(self.draw_projection)
        boxLayout.addWidget(drawButton)

        groupBox = QGroupBox("Druggability fingerprint")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotsExpressionLine = QLineEdit("")
        boxLayout.addRow("Hotspots:", self.hotspotsExpressionLine)

        self.refSiteExpressionLine = QLineEdit("")
        boxLayout.addRow("Site:", self.refSiteExpressionLine)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(4)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.nBinsSpin = QSpinBox()
        self.nBinsSpin.setValue(5)
        self.nBinsSpin.setMinimum(0)
        self.nBinsSpin.setMaximum(50)
        boxLayout.addRow("Fingerprint bins:", self.nBinsSpin)

        self.fingerprintsCheck = QCheckBox()
        self.fingerprintsCheck.setChecked(True)
        boxLayout.addRow("Fingerprints:", self.fingerprintsCheck)

        self.dendrogramCheck = QCheckBox()
        self.dendrogramCheck.setChecked(False)
        boxLayout.addRow("Dendrogram:", self.dendrogramCheck)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_fingerprint)
        boxLayout.addWidget(plotButton)

    def draw_projection(self):
        hotspots = self.hotspotsExpressionLine.text()
        protein = self.proteinExpressionLine.text()
        radius = self.radiusSpin.value()
        type = self.typeCombo.currentText()
        palette = self.paletteLine.text()

        hs_proj(hotspots, protein, radius, type, palette)

    def plot_fingerprint(self):
        hotspots = self.hotspotsExpressionLine.text()
        ref_site = self.refSiteExpressionLine.text()
        radius = self.radiusSpin.value()
        fingerprints = self.fingerprintsCheck.isChecked()
        dendrogram = self.dendrogramCheck.isChecked()
        nbins = self.nBinsSpin.value()

        fp_sim(
            hotspots,
            ref_site,
            radius,
            verbose=True,
            plot_fingerprints=fingerprints,
            plot_dendrogram=dendrogram,
            nbins=nbins,
        )

class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("XDrugPy")

        tab = QTabWidget()
        tab.addTab(LoadWidget(), "Load")
        tab.addTab(TableWidget(), "Properties")
        tab.addTab(SimilarityWidget(), "Hotspot Similarity")
        tab.addTab(CountWidget(), "Probe Count")

        layout.addWidget(tab)


dialog = None


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = MainDialog()
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("XDrugPy", run_plugin_gui)


# if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
#     run_plugin_gui()