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
import shutil
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
    import seaborn as sb
    import pandas as pd
    import openpyxl
    from scipy.spatial import distance_matrix, distance
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import pearsonr
    from matplotlib import pyplot as plt
    from matplotlib import ticker
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
import json
import re
from enum import Enum
from fnmatch import fnmatch
from glob import glob
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from functools import lru_cache

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib
from scipy.spatial import distance_matrix, distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from pymol import cmd as pm, parsing
from matplotlib import pyplot as plt
from matplotlib import ticker
from strenum import StrEnum

__all__ = [
    "load_ftmap",
    "fo",
    "dc",
    "dce",
    "jaccard",
    "fp_sim",
    "ho",
    "res_sim",
    "hs_proj",
    "plot_dendrogram",
    "plot_heatmap",
]

matplotlib.use("Qt5Agg")


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


def get_kozakov2015(group, clusters, max_length=8):
    k15 = []
    for length in range(1, max_length + 1):
        for combination in combinations(clusters, length):
            cd = []
            cluster1 = combination[0]
            avg1 = np.average(cluster1.coords, axis=0)
            for cluster2 in combination:
                avg2 = np.average(cluster2.coords, axis=0)
                cd.append(distance.euclidean(avg1, avg2))

            coords = np.concatenate([c.coords for c in combination])
            hs = SimpleNamespace(
                selection=" or ".join(c.selection for c in combination),
                clusters=combination,
                kozakov_class=None,
                strength=sum(c.strength for c in combination),
                strength0=combination[0].strength,
                center_center=np.max(cd),
                max_dist=distance_matrix(coords, coords).max(),
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

    k15 = sorted(k15, key=lambda hs: (-hs.strength0, hs.strength))
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
    # with tempfile.TemporaryDirectory() as tempdir:
    tempdir = "/tmp"
    if True:
        protein_pdb = f"{tempdir}/{group}.pdb"
        pm.save(protein_pdb, selection=protein)
        subprocess.check_call(
            [fpocket_bin, "-f", protein_pdb],
            env={
                "TMPDIR": QStandardPaths.writableLocation(QStandardPaths.TempLocation)
            },
        )
        header_re = re.compile(r"^HEADER\s+\d+\s+-(.*):(.*)$")
        for pocket_pdb in glob(f"{tempdir}/{group}_out/pockets/pocket*_atm.pdb"):
            idx = (
                os.path.basename(pocket_pdb)
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
    k15_list = get_kozakov2015(group, clusters)
    fpo_list = get_fpocket(group, f"{group}.protein")
    process_clusters(group, clusters)
    process_eclusters(group, eclusters)
    e19_list = get_egbert2019(group, fpo_list, clusters)

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
    state1: int = 1,
    state2: int = 1,
    verbose: bool = True,
):
    """
    Compute the fractional overlap of sel1 respective to sel2.
        FO = Nc/Nt

    Nc is the number of atoms of sel1 in contact with sel2. Nt is the number of atoms
    of sel1. Hydrogen atoms are ignored.

    Default to first state from sel1 and sel2.

    OPTIONS
        sel1    ligand object.
        sel2    hotspot object.
        radius  the radius so sel1 and sel2 are in contact (default: 2).
        state1  state of sel1.
        state2  state of sel2.

    EXAMPLE
        fo REF_LIG, ftmap1234.D_003_*_*
    """
    atoms1 = pm.get_coords(f"({sel1}) and not elem H", state=state1)
    atoms2 = pm.get_coords(f"({sel2}) and not elem H", state=state2)
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
    state1: int = 1,
    state2: int = 1,
    verbose: bool = True,
):
    """
    Compute the Density Correlation according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are the selections representing the molecules or hotspots;
    state1 and state2 are the optional corresponding states (default to first
    state both). The threshold distance can be changed with radius.

    OPTIONS
        sel1    first object
        sel2    second object
        radius  the radius so two atoms are in contact (default: 1.25)
        state1  state of sel1
        state2  state of sel2
        verbose define verbosity

    EXAMPLES
        dc REF_LIG, ftmap1234.D_003_*_*
        dc ftmap1234.D.003, REF_LIG, radius=1.5

    """
    xyz1 = pm.get_coords(f"({sel1}) and not elem H", state1)
    xyz2 = pm.get_coords(f"({sel2}) and not elem H", state2)

    dc_ = (distance_matrix(xyz1, xyz2) < radius).sum()
    if verbose:
        print(f"DC: {dc_:.2f}")
    return dc_


@declare_command
def dce(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    state1: int = 1,
    state2: int = 1,
    verbose: bool = True,
):
    """
    Compute the Density Correlation Efficiency according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are respectively the molecule and hotspot; state1 and state2 are
    the optional corresponding states (default to first state both). The threshold
    distance can be changed with radius.

    OPTIONS
        sel1    ligand object
        sel2    hotspot object
        radius  the radius so two atoms are in contact (default: 1.25)
        state1  state of sel1
        state2  state of sel2
        verbose define verbosity

    EXAMPLE
        dce REF_LIG, ftmap1234.D_003_*_*
    """
    dc_ = dc(sel1, sel2, radius, state1, state2, verbose=False)
    dce_ = dc_ / pm.count_atoms(f"({sel1}) and not elem H")
    if verbose:
        print(f"DCE: {dce_:.2f}")
    return dce_


@lru_cache(maxsize=1_000)
@declare_command
def jaccard(
    sel1: Selection,
    sel2: Selection,
    state1: int = 1,
    state2: int = 1,
    verbose: bool = True,
):
    """
    Compute the Jaccard similarity index of residues between sel1 and sel2. Two
    residues are considered the same if they match chain and resi after cealign.

    OPTIONS:
        sel1	a selection
        sel2	another selection
        verbose define verbosity

    EXAMPLES
        jaccard polymer within 4 of *.D_003_*_*, polymer within *.D_002_*_*
        jaccard polymer within 4 of *.B_*, polymer within *.D.*

    """

    site = [(a.chain, a.resi) for a in pm.get_model(sel2).atom]
    try:
        aln_obj = pm.get_unused_name()
        pm.cealign(
            f"bymolecule {sel2}", f"bymolecule {sel1}", transform=0, object=aln_obj
        )
        raw = pm.get_raw_alignment(aln_obj)

        resis = {}
        pm.iterate(
            aln_obj, "resis[model, index] = (chain, resi)", space={"resis": resis}
        )

        resis2 = set()
        for idx1, idx2 in raw:
            if resis[idx1] in site:
                resis2.add(resis[idx2])

        resis1 = set()
        pm.iterate(sel1, "resis1.add((chain, resi))", space={"resis1": resis1})

        try:
            ret = len(resis1.intersection(resis2)) / len(resis1.union(resis2))
        except ZeroDivisionError:
            print("Your selections yields zero atoms.")
            return 0.0

        if verbose:
            print(f"Jaccard similarity: {ret:.2}")
        return ret
    finally:
        pm.delete(aln_obj)


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"


@declare_command
def fp_sim(
    hss: Selection,
    ref_site: str = "*",
    radius: int = 3,
    verbose: bool = True,
    plot_fingerprints: bool = True,
    plot_fingerprints_nbins: int = 10,
    plot_dendrogram: bool = True,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
):
    """
    Compute the similarity between the residue contact fingerprint of two
    hotspots.

    OPTIONS:
        hss         hotspot expressions
        protein     protein objects
        ref_site    selection based on first protein to focus
        radius      radius to compute the contacts (default: 3)
        verbose     define verbosity
        plot        plot the fingerprints (default: True)
        max_bins    maximum number of residue labels for the fingerprint (default: 10)

    EXAMPLES
        fs_sim 8DSU.D_001*, 6XHM.D_001*
        fs_sim 8DSU.CS_*, 6XHM.CS_*, ref_site=resi 8-101, max_bins=20
        fp_sim (bm. 8DSU.D_*) within 2.5 of LIG, \\
               (bm. 6XHM.D_*) within 2.5 of LIG, \\
               protein1=8DSU.protein, \\
               protein2=6XHM.protein \\
               radius=3
    """

    expanded_hss = []
    all_groups = [g.split(".", maxsplit=1)[0] for g in pm.get_object_list("*.protein")]
    for expr in hss.split():
        expr_g, expr_part = expr.split(".", maxsplit=1)
        for g in all_groups:
            if fnmatch(g, expr_g):
                expanded_hss.append("%s.%s" % (g, expr_part))

    hss = expanded_hss
    groups = [hs.split(".", maxsplit=1)[0] for hs in hss]
    proteins = [f"{g}.protein" for g in groups]

    p0 = proteins[0]
    titles = [p[:-8] if p.endswith(".protein") else p for p in proteins]

    fp_list = [None] * len(proteins)
    p_list = []

    plt.close()
    fig = plt.figure()
    fig_nrows = 0

    if plot_dendrogram:
        fig_nrows += 1
    if plot_fingerprints:
        fig_nrows += len(proteins)

    if plot_fingerprints and plot_dendrogram:
        dendro_ax = fig.add_subplot(fig_nrows, 1, 1)
        fp_axs = []
        for i, p in enumerate(proteins):
            fp_axs.append(fig.add_subplot(fig_nrows, 1, i + 2))

    if plot_fingerprints and not plot_dendrogram:
        fp_axs = []
        for i, p in enumerate(proteins):
            fp_axs.append(fig.add_subplot(fig_nrows, 1, i + 1))

    if not plot_fingerprints and plot_dendrogram:
        dendro_ax = fig.add_subplot(fig_nrows, 1, 1)

    for i1, (hs1, p1) in enumerate(zip(hss, proteins)):
        for i2, (hs2, p2) in enumerate(zip(hss, proteins)):
            if i1 >= i2:
                continue
            try:
                aln_obj = pm.get_unused_name()
                pm.cealign(
                    f"bymolecule {p1}", f"bymolecule {p2}", transform=0, object=aln_obj
                )
                raw = pm.get_raw_alignment(aln_obj)

                resis = {}
                pm.iterate(
                    aln_obj,
                    "resis[model, index] = (segi, chain, resi, resn)",
                    space={"resis": resis},
                )
            finally:
                pm.delete(aln_obj)

            site = []
            if ref_site:
                pm.iterate(
                    f"(bymolecule {p1}) and name CA and ({ref_site})",
                    "site.append(index)",
                    space={"site": site},
                )

            lbl1 = []
            lbl2 = []
            fp1 = []
            fp2 = []
            for (model1, index1), (model2, index2) in raw:
                if ref_site and index1 not in site:
                    continue
                segi1, chain1, resi1, resn1 = resis[(model1, index1)]
                segi2, chain2, resi2, resn2 = resis[(model2, index2)]
                cnt1 = count_molecules(
                    f"({hs1}) within {radius} from /{model1}/{segi1}/{chain1}/{resi1}"
                )
                cnt2 = count_molecules(
                    f"({hs2}) within {radius} from /{model2}/{segi2}/{chain2}/{resi2}"
                )
                fp1.append(cnt1)
                fp2.append(cnt2)

                lbl1.append(f"{resn1}_{resi1}_{chain1}")
                lbl2.append(f"{resn2}_{resi2}_{chain2}")

            p = pearsonr(fp1, fp2).statistic
            if np.isnan(p):
                p = 0.0
            p_list.append(p)

            if verbose:
                print(f" Pearson correlation ({hs1}, {hs2}) = {p:.2}")

            if plot_fingerprints:
                for i, fp, hs, lbl in [(i1, fp1, hs1, lbl1), (i2, fp2, hs2, lbl2)]:
                    if fp_list[i] is not None:
                        continue
                    fp_list[i] = fp
                    ax = fp_axs[i]

                    ax.set_ylabel(hs)
                    ax.bar(np.arange(len(fp)), fp)
                    ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
                    ax.set_xticks(np.arange(len(fp)), labels=lbl, rotation=45)
                    ax.locator_params(
                        axis="x", tight=True, nbins=plot_fingerprints_nbins
                    )
                    for label in ax.xaxis.get_majorticklabels():
                        label.set_horizontalalignment("right")

    if plot_dendrogram:
        dendrogram(
            linkage([1 - p for p in p_list], method="average"),
            labels=hss,
            ax=dendro_ax,
            leaf_rotation=45,
            color_threshold=0,
        )
    plt.tight_layout()
    plt.show()
    return p_list


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


@declare_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: int = 4,
    verbose: bool = True,
):
    """
    Compute hotspots by the Jaccard similarity of nearby residues.

    OPTIONS
        hs1     an hotspot object
        hs2     another hotspot object
        radius  the distance to consider residues near hotspots (default: 4)
        verbose define verbosity

    EXAMPLES
        res_sim 8DSU.D_001*, 6XHM.D_001*
        res_sim 8DSU.CS_*, 6XHM.CS_*
    """

    group1 = hs1.split(".", maxsplit=1)[0]
    group2 = hs2.split(".", maxsplit=1)[0]

    sel1 = f"{group1}.protein within {radius} from ({hs1})"
    sel2 = f"{group2}.protein within {radius} from ({hs2})"

    try:
        ret = jaccard(sel1, sel2, verbose=verbose)
    except:
        raise Exception(f"Cannot compute residue_similarity of {hs1} and {hs2}")

    return ret


class CrossMeasureFunction(StrEnum):
    DC = "dc"
    FO = "fo"
    HO = "ho"
    RESIDUE = "res_sim"


@declare_command
def plot_heatmap(
    objs: Selection,
    function: CrossMeasureFunction = CrossMeasureFunction.FO,
    radius: float = 2.0,
    annotate: bool = False,
):
    """
    Compute the similarity between matching objects using a similarity function.

    OPTIONS
        objs        space separated list of object expressions
        function    dce, fo, or residue
        radius      the radius to consider atoms in contact (default: 2.0)
        annotate    fill the cells with values

    EXAMPLES
        cross_measure *.D_000_*_*, function=dce
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
        return klass, obj

    obj1s = list(sorted(obj1s, key=sort))

    if len(obj1s) == 0:
        raise ValueError("No objects found")

    mat = []
    for idx1, obj1 in enumerate(obj1s):
        mat.append([])
        for idx2, obj2 in enumerate(obj1s):
            match function:
                case CrossMeasureFunction.DC:
                    ret = dc(obj1, obj2, radius=radius, verbose=False)

                case CrossMeasureFunction.FO:
                    ret = fo(obj1, obj2, radius=radius, verbose=False)

                case CrossMeasureFunction.RESIDUE:
                    ret = res_sim(obj1, obj2, verbose=False)

                case CrossMeasureFunction.HO:
                    ret = ho(obj1, obj2, radius=radius, verbose=False)

            mat[-1].append(round(ret, 2))

    if function == CrossMeasureFunction.DC:
        vmax = np.max(mat)
    else:
        vmax = 1

    plt.close()

    fig, ax = plt.subplots(1)
    sb.heatmap(
        mat,
        yticklabels=obj1s,
        xticklabels=obj1s,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
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
                j = res_sim(obj1, obj2, verbose=False)
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


if __name__ in ["pymol", "pmg_tk.startup.XDrugPy"]:
    import pymol

    QWidget = pymol.Qt.QtWidgets.QWidget
    QFileDialog = pymol.Qt.QtWidgets.QFileDialog
    QFormLayout = pymol.Qt.QtWidgets.QFormLayout
    QPushButton = pymol.Qt.QtWidgets.QPushButton
    QSpinBox = pymol.Qt.QtWidgets.QSpinBox
    QDoubleSpinBox = pymol.Qt.QtWidgets.QDoubleSpinBox
    QLineEdit = pymol.Qt.QtWidgets.QLineEdit
    QCheckBox = pymol.Qt.QtWidgets.QCheckBox
    QVBoxLayout = pymol.Qt.QtWidgets.QVBoxLayout
    QHBoxLayout = pymol.Qt.QtWidgets.QHBoxLayout
    QDialog = pymol.Qt.QtWidgets.QDialog
    QComboBox = pymol.Qt.QtWidgets.QComboBox
    QTabWidget = pymol.Qt.QtWidgets.QTabWidget
    QLabel = pymol.Qt.QtWidgets.QLabel
    QTableWidget = pymol.Qt.QtWidgets.QTableWidget
    QTableWidgetItem = pymol.Qt.QtWidgets.QTableWidgetItem
    QGroupBox = pymol.Qt.QtWidgets.QGroupBox
    QHeaderView = pymol.Qt.QtWidgets.QHeaderView

    QtCore = pymol.Qt.QtCore
    QIcon = pymol.Qt.QtGui.QIcon

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
                                raise ValueError(f"Invalid filename '{filename}'")
                            else:
                                raise Exception(f"Could not load file '{filename}'")
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
            self.functionCombo.addItems([e.value for e in CrossMeasureFunction])
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
            self.radiusSpin.setValue(3)
            self.radiusSpin.setMinimum(2)
            self.radiusSpin.setMaximum(5)
            boxLayout.addRow("Radius:", self.radiusSpin)

            self.nBinsSpin = QSpinBox()
            self.nBinsSpin.setValue(15)
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
                plot_fingerprints_nbins=nbins,
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
