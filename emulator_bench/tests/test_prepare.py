"""prepare_splits.py on a synthetic dataset: one structure per sequence, rows without a structure stay."""

import json
import sys
from datetime import datetime

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import pytest

from conftest import random_coords, write_pdb
from emulator_bench import prepare_splits
from emulator_bench.feature_utils import chain_residues

SEQUENCES = {"S1": "MKVLAA", "S2": "MKVLAG", "S3": "MKVLAC", "S4": "MKVLAD", "S5": "MKVLAE"}


def write_cif(path, chains):
    """An mmCIF file of ALA residues. chains maps a chain id to (first residue number, residue count, coordinate seed)."""
    atoms = []
    for chain, (first, count, seed) in chains.items():
        coords = random_coords(count * 3, seed)
        for i in range(count * 3):
            atoms.append(struc.Atom(coords[i], chain_id=chain, res_id=first + i // 3, res_name="ALA", atom_name=("N", "CA", "C")[i % 3], element="NCC"[i % 3], hetero=False))
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, struc.array(atoms), include_bonds=False)
    cif.write(path)


@pytest.fixture
def dataset(tmp_path):
    """Returns (task dir, structure root). The candidate table lists several rows for S1 and none for S5."""
    structures = tmp_path / "structures"
    for name in ("mmcif", "alphafold", "esmfold2_pdb", "esm"):
        (structures / name).mkdir(parents=True)
    write_cif(structures / "mmcif" / "1ABC_updated.cif", {"A": (1, 9, 1), "B": (5, 6, 2)})
    write_cif(structures / "mmcif" / "2DEF.cif", {"A": (1, 6, 3)})
    write_pdb(structures / "alphafold" / "AF-P2-F1-model_v1.pdb", random_coords(6, 4))
    write_pdb(structures / "alphafold" / "AF-P2-F1-model_v2.pdb", random_coords(6, 5))
    write_pdb(structures / "esmfold2_pdb" / "ESMFold2-P3|A1B.pdb", random_coords(6, 6))
    write_pdb(structures / "esm" / "ESM3-open-small-P4.pdb", random_coords(6, 7))

    task = tmp_path / "data" / "kcat"
    candidate = {"uniprot_id": None, "pdbs": None, "pdb_type": "predicted", "pdb_chain": None, "pdb_coverage": np.nan}
    rows = [
        {**candidate, "sequence": SEQUENCES["S1"], "uniprot_id": "P1", "pdbs": "2DEF:P1", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.9},
        {**candidate, "sequence": SEQUENCES["S1"], "uniprot_id": "P1", "pdbs": "1ABC:P1", "pdb_type": "experimental", "pdb_chain": "B", "pdb_coverage": 0.9},
        {**candidate, "sequence": SEQUENCES["S1"], "uniprot_id": "P1", "pdbs": "1ABC:P1", "pdb_type": "experimental", "pdb_chain": "A", "pdb_coverage": 0.2},
        {**candidate, "sequence": SEQUENCES["S2"], "uniprot_id": "P2", "pdbs": "P2"},
        {**candidate, "sequence": SEQUENCES["S3"], "uniprot_id": "P3|A1B", "pdbs": "P3|A1B"},
        {**candidate, "sequence": SEQUENCES["S4"], "uniprot_id": "P4", "pdbs": "P4"},
    ]
    (task / "random_splits_grouped_sequence").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(task / "kcat_kinetic_params_3d.parquet")
    parts = {"train": ["S1", "S2", "S3"], "val": ["S4"], "test": ["S5", "S1"]}
    for part, names in parts.items():
        pd.DataFrame(
            {
                "sequence": [SEQUENCES[n] for n in names],
                "smiles": "CCO",
                "value": 2.0,
                "smiles_hash": "h",
                "uniprot_date": datetime(2020, 1, 1),
                "log10_value": 0.3,
            }
        ).to_parquet(task / "random_splits_grouped_sequence" / f"{part}.parquet")
    return task, structures


def run_prepare(monkeypatch, tmp_path, task, structures, *extra):
    argv = ["prepare_splits.py", "--base_dir", str(task), "--manifests_dir", str(tmp_path / "manifests"), "--chains_dir", str(tmp_path / "chains"),
            "--structure_root", str(structures), "--split_groups", "random_splits_grouped_sequence", "--workers", "1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    prepare_splits.main()
    return json.loads((tmp_path / "manifests" / "prepare_summary.json").read_text())


def manifest(tmp_path, part):
    return pd.read_parquet(tmp_path / "manifests" / "random_splits_grouped_sequence" / "all" / f"{part}.parquet")


def test_one_structure_per_sequence_and_no_structure_rows_stay(dataset, tmp_path, monkeypatch):
    task, structures = dataset
    summary = run_prepare(monkeypatch, tmp_path, task, structures)
    train, val, test = (manifest(tmp_path, part) for part in ("train", "val", "test"))
    assert train["structure_source"].tolist() == ["experimental", "alphafold", "esmfold2"]
    assert val["structure_source"].tolist() == ["esm3"]
    assert test["structure_source"].tolist() == ["none", "experimental"]
    assert train["structure_path"][0] == test["structure_path"][1] == str(tmp_path / "chains" / "1ABC_B.pdb")  # tie: first candidate in sort order
    assert train["structure_path"][1].endswith("AF-P2-F1-model_v2.pdb")
    assert train["structure_id"][2] == "P3|A1B"
    assert test["structure_path"][0] == ""
    assert len(test) == 2 and (train["target"] == 0.3).all()
    counts = {(row["partition"]): row for row in summary["splits"]}
    assert (counts["test"]["n_rows"], counts["test"]["none"], counts["test"]["experimental"]) == (2, 1, 1)


def test_extracted_chain_is_the_author_chain_renumbered(dataset, tmp_path, monkeypatch):
    task, structures = dataset
    run_prepare(monkeypatch, tmp_path, task, structures)
    chain = tmp_path / "chains" / "1ABC_B.pdb"
    letters, coords = chain_residues(chain)
    assert letters == "AAAAAA"
    np.testing.assert_allclose(coords, random_coords(18, 2)[1::3], atol=1e-3)
    assert chain.read_text().count(" CA ") == 6 and "A   1" in chain.read_text() and "A   6" in chain.read_text()
    assert not (tmp_path / "chains" / "1ABC_A.pdb").exists()


def test_rerun_skips_and_a_new_structure_rewrites(dataset, tmp_path, monkeypatch):
    task, structures = dataset
    run_prepare(monkeypatch, tmp_path, task, structures)
    assert {row["status"] for row in run_prepare(monkeypatch, tmp_path, task, structures)["splits"]} == {"skipped_exists"}
    write_pdb(structures / "esmfold2_pdb" / "ESMFold2-P2.pdb", random_coords(6, 8))  # AlphaFold still wins for S2
    assert {row["status"] for row in run_prepare(monkeypatch, tmp_path, task, structures)["splits"]} == {"skipped_exists"}
    (structures / "alphafold" / "AF-P5-F1-model_v1.pdb").write_text((structures / "esm" / "ESM3-open-small-P4.pdb").read_text())
    candidates = pd.read_parquet(task / "kcat_kinetic_params_3d.parquet")
    extra = candidates.iloc[[0]].assign(sequence=SEQUENCES["S5"], uniprot_id="P5", pdbs="P5", pdb_type="predicted", pdb_chain=None)
    pd.concat([candidates, extra]).to_parquet(task / "kcat_kinetic_params_3d.parquet")
    statuses = {row["partition"]: row["status"] for row in run_prepare(monkeypatch, tmp_path, task, structures)["splits"]}
    assert statuses == {"train": "skipped_exists", "val": "skipped_exists", "test": "written"}
    assert manifest(tmp_path, "test")["structure_source"].tolist() == ["alphafold", "experimental"]
