"""Feature functions against Code/Example/example.py, and the feature cache."""

import copy
import pickle
import shutil
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from conftest import DICT_DIR, random_coords, write_pdb
from emulator_bench import cache_features, feature_utils
from emulator_bench.cache_features import build_feature_split, load_dictionaries, substrate_features

SMILES = ["CCO", "c1ccccc1O", "CC(=O)O", "CN(CC(=O)O)C(=N)N", "C1CCCCC1", "OC[C@H](O)C=O"]
SEQUENCES = ["MKVLAAGIVGLLLAQPAMA", "MPFGNTHNKFKLNYKPEEEYPDLSKHNNHMAKVLTLELYKKLRDKETPSGFTVDDVIQTGVDNPGHPFIMT", "XXXXBBBZ"]


@pytest.fixture(scope="module")
def dicts():
    return load_dictionaries(DICT_DIR)


def original_substrate(example, smiles, dicts, radius=2):
    """The substrate lines of example.py main(), on copies of the dictionaries (example.py adds missing keys to them)."""
    fingerprint_dict, edge_dict = copy.deepcopy(dicts["fingerprint_dict"]), copy.deepcopy(dicts["edge_dict"])
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms = example.create_atoms(mol, dicts["atom_dict"])
    bonds = example.create_ijbonddict(mol, dicts["bond_dict"])
    fingerprints = example.extract_fingerprints(atoms, bonds, radius, fingerprint_dict, edge_dict)
    return fingerprints, example.create_adjacency(mol)


@pytest.mark.parametrize("smiles", SMILES)
def test_substrate_features_match_example(cpu_original, dicts, smiles):
    expected_fp, expected_adj = original_substrate(cpu_original.example, smiles, dicts)
    fp, adj = substrate_features(smiles, Namespace(radius=2), dicts)
    np.testing.assert_array_equal(fp, expected_fp)
    np.testing.assert_array_equal(adj, expected_adj)


REPAIRED = {
    "[Na+].[Cl-]": [0, 0],  # atoms without a bond
    "[H].[H]": [0, 0],
    "[H]": [0],  # single atom without a fingerprint entry
    "[I-]": [0],
    "[Cu+2]": [0],  # unknown atom
    "CCC(=O)SCC[N+](C)(C)C.[I-]": None,
    "C[As+](C)(C)CCCC(=O)O": None,
}


@pytest.mark.parametrize("smiles", REPAIRED)
def test_failing_substrates_are_repaired(cpu_original, dicts, smiles):
    with pytest.raises((KeyError, IndexError, AssertionError)):
        fp, adj = original_substrate(cpu_original.example, smiles, dicts)
        assert len(fp) == len(adj)
    before = [len(dicts[k]) for k in ("atom_dict", "bond_dict", "fingerprint_dict", "edge_dict")]
    fp, adj = substrate_features(smiles, Namespace(radius=2), dicts)
    assert len(fp) == len(adj)
    if REPAIRED[smiles] is not None:
        assert fp.tolist() == REPAIRED[smiles]
    assert before == [len(dicts[k]) for k in ("atom_dict", "bond_dict", "fingerprint_dict", "edge_dict")]


def test_unknown_atom_makes_its_fingerprints_unseen(dicts):
    fp, _ = substrate_features("C[As+](C)(C)CCCC(=O)O", Namespace(radius=2), dicts)
    assert fp[:4].tolist() == [0, 0, 0, 0]  # the As atom and its three neighbours


def test_isolated_atoms_keep_the_fingerprints_of_the_bonded_part(cpu_original, dicts):
    """CCO with an extra ion: the ethanol fingerprints are those of ethanol alone, in atom order."""
    ethanol, _ = substrate_features("CCO", Namespace(radius=2), dicts)
    both, _ = substrate_features("[Na+].CCO", Namespace(radius=2), dicts)
    assert both[0] == 0 and both[1:].tolist() == ethanol.tolist()


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_sequence_tokens_match_example(cpu_original, dicts, sequence):
    expected = cpu_original.example.split_sequence(sequence, 4, copy.deepcopy(dicts["word_dict"]))
    np.testing.assert_array_equal(feature_utils.split_sequence(sequence, 4, dicts["word_dict"]), expected)


def contact_pair(example, pdb, seq):
    expected = example.luciferase_contact_map(str(pdb), seq)
    ours, n_aligned, n_match = feature_utils.protein_contact_map(str(pdb), seq)
    return expected, ours, n_aligned


@pytest.mark.parametrize("n_ca, length", [(40, 40), (40, 47), (1, 5)])
def test_contact_map_matches_example_without_gaps(cpu_original, tmp_path, n_ca, length):
    """A chain that is a prefix of the sequence gives the original map, including its padding."""
    pdb = tmp_path / "chain.pdb"
    write_pdb(pdb, random_coords(n_ca, 7))
    expected, ours, n = contact_pair(cpu_original.example, pdb, "A" * n_ca + "G" * (length - n_ca))
    assert n == n_ca
    np.testing.assert_array_equal(ours.toarray(), expected.toarray())
    assert ours.dtype == expected.dtype if n_ca < length else True


def test_contact_map_of_the_example_structure(cpu_original):
    pdb = DICT_DIR / "2.7.3.2_Homo_sapiens_Seq_5476_relaxed_rank_1_model_3.pdb"
    letters, _ = feature_utils.chain_residues(pdb)
    for seq in (letters, letters + "G" * 6):
        expected, ours, n = contact_pair(cpu_original.example, pdb, seq)
        assert n == len(letters)
        np.testing.assert_array_equal(ours.toarray(), expected.toarray())


SEQ = "MKVLAAGIVGLLLAQPAMAWQERTYHNDCSF"  # I (7) and W (19) occur once, so a deleted I or a W mutation aligns without ambiguity


def hand_map(length, positions, coords, threshold=10.0):
    """Expected map: contacts between the placed C-alpha atoms, a unit diagonal everywhere."""
    expected = np.eye(length)
    for a, pa in enumerate(positions):
        for b, pb in enumerate(positions):
            expected[pa, pb] = max(expected[pa, pb], np.linalg.norm(coords[a] - coords[b]) < threshold)
    return expected


def aligned_map(tmp_path, chain_seq, seq, seed=5, spread=12.0):
    coords = random_coords(len(chain_seq), seed, spread)
    pdb = tmp_path / "chain.pdb"
    write_pdb(pdb, coords, sequence=chain_seq)
    return coords, *feature_utils.protein_contact_map(str(pdb), seq)


def test_gap_in_the_chain_leaves_the_position_without_contacts(tmp_path):
    chain_seq = SEQ[:7] + SEQ[8:]
    coords, ours, n_aligned, n_match = aligned_map(tmp_path, chain_seq, SEQ)
    positions = [i for i in range(len(SEQ)) if i != 7]
    np.testing.assert_array_equal(ours.toarray(), hand_map(len(SEQ), positions, coords))
    assert (n_aligned, n_match) == (len(SEQ) - 1, len(SEQ) - 1)
    assert ours.dtype == np.float64 and ours.toarray()[7].sum() == 1


def test_n_terminal_tag_is_cut(tmp_path):
    tag = "GSHMGS"
    coords, ours, n_aligned, n_match = aligned_map(tmp_path, tag + SEQ, SEQ)
    np.testing.assert_array_equal(ours.toarray(), hand_map(len(SEQ), range(len(SEQ)), coords[len(tag):]))
    assert (n_aligned, n_match) == (len(SEQ), len(SEQ)) and ours.dtype == np.int32


def test_point_mutation_stays_aligned(tmp_path):
    chain_seq = SEQ[:19] + "G" + SEQ[20:]
    coords, ours, n_aligned, n_match = aligned_map(tmp_path, chain_seq, SEQ)
    np.testing.assert_array_equal(ours.toarray(), hand_map(len(SEQ), range(len(SEQ)), coords))
    assert (n_aligned, n_match) == (len(SEQ), len(SEQ) - 1)


def test_residues_from_1000_are_read(tmp_path):
    seq = "".join(np.random.RandomState(0).choice(list("ACDEFGHIKLMNPQRSTVWY"), 1005))
    coords, ours, n_aligned, _ = aligned_map(tmp_path, seq, seq, spread=60.0)
    assert n_aligned == 1005
    np.testing.assert_array_equal(ours.toarray(), hand_map(1005, range(1005), coords))
    gapped = seq[:500] + seq[501:]
    coords, ours, n_aligned, _ = aligned_map(tmp_path, gapped, seq, spread=60.0)
    assert n_aligned == 1004
    np.testing.assert_array_equal(ours.toarray(), hand_map(1005, [i for i in range(1005) if i != 500], coords))


def test_chain_residues_reads_fixed_columns(tmp_path):
    pdb = tmp_path / "wide.pdb"
    write_pdb(pdb, random_coords(3, 1), first_residue=998, sequence="MKV")
    letters, coords = feature_utils.chain_residues(pdb)
    assert letters == "MKV" and coords.shape == (3, 3)


def feature_args(tmp_path, **overrides):
    values = {"radius": 2, "ngram": 4, "dist_thres": 10.0, "limit_rows": None, "overwrite": False, "fail_on_error": False, "cache_dir": tmp_path / "cache"}
    return Namespace(**{**values, **overrides})


@pytest.fixture
def manifest(tmp_path):
    seq = SEQ[:8]
    chains = {
        "equal": seq,
        "short": seq[:5],
        "tagged": "GSHM" + seq + "GGGGGG",
        "mutated": seq[:2] + "WW" + seq[4:],
    }
    paths = {}
    for i, (name, chain_seq) in enumerate(chains.items()):
        paths[name] = tmp_path / f"{name}.pdb"
        write_pdb(paths[name], random_coords(len(chain_seq), i), sequence=chain_seq)
    rows = [
        (seq, "CCO", paths["equal"], "experimental"),
        (seq, "CC(=O)O", paths["short"], "alphafold"),
        (seq, "c1ccccc1O", "", "none"),
        (seq, "not a smiles", paths["equal"], "experimental"),
        (seq, "[Na+].[Cl-]", paths["equal"], "experimental"),
        (seq, "CCO", paths["tagged"], "esm3"),
        (seq, "CCO", paths["mutated"], "esm3"),
    ]
    frame = pd.DataFrame(rows, columns=["sequence", "smiles", "structure_path", "structure_source"])
    frame["structure_path"] = frame["structure_path"].astype(str)
    frame["value"] = 2.0
    frame["target"] = 1.0
    path = tmp_path / "manifest.parquet"
    frame.to_parquet(path)
    return path


def test_build_feature_split_counts(manifest, tmp_path, dicts):
    out = tmp_path / "features"
    summary = build_feature_split(manifest, out, feature_args(tmp_path), dicts, cache_features.dictionary_hashes(DICT_DIR))
    assert (summary["rows_in"], summary["rows_out"], summary["rows_failed"]) == (7, 6, 1)
    assert summary["failed_by_stage"] == {"substrate": 1}
    structure = summary["structure"]
    assert structure.pop("mean_coverage") == pytest.approx((1 + 5 / 8 + 1 + 1 + 1) / 5)
    assert structure == {
        "n_rows": 6,
        "n_no_structure": 1,
        "n_aligned_full": 4,
        "n_partial": 1,
        "n_low_identity": 1,  # the chain with two wrong letters out of eight pairs
        "sources": {"experimental": 2, "alphafold": 1, "esmfold2": 0, "esm3": 2, "none": 1},
    }
    maps = np.load(out / "proteinadjacencies.npy", allow_pickle=True)
    assert [m.shape for m in maps] == [(8, 8)] * 6
    np.testing.assert_array_equal(maps[2].toarray(), np.eye(8))  # no structure: self-loops only
    np.testing.assert_array_equal(np.load(out / "logkcat.npy", allow_pickle=True)[0], [1.0])  # log2(2)


def test_cache_hits_do_not_change_the_counts(manifest, tmp_path, dicts):
    hashes = cache_features.dictionary_hashes(DICT_DIR)
    first = build_feature_split(manifest, tmp_path / "a", feature_args(tmp_path), dicts, hashes)
    second = build_feature_split(manifest, tmp_path / "b", feature_args(tmp_path), dicts, hashes)
    assert second["structure_hits"] == 5 and second["structure_writes"] == 0
    assert second["structure"] == first["structure"]
    assert second["failed_by_stage"] == first["failed_by_stage"]


def changed_dictionaries(tmp_path):
    folder = tmp_path / "other_dicts"
    shutil.copytree(DICT_DIR, folder, ignore=shutil.ignore_patterns("*.pdb", "*.npy"))
    with open(folder / "sequence_dict_0612.pickle", "rb") as handle:
        words = pickle.load(handle)
    words["ZZZZ"] = len(words)
    with open(folder / "sequence_dict_0612.pickle", "wb") as handle:
        pickle.dump(words, handle)
    return folder


@pytest.mark.parametrize("change", ["same", "radius", "ngram", "dist_thres", "dictionary", "resolver_version"])
def test_cache_key_covers_the_configuration(manifest, tmp_path, dicts, monkeypatch, change):
    hashes = cache_features.dictionary_hashes(DICT_DIR)
    out = tmp_path / "features"
    assert build_feature_split(manifest, out, feature_args(tmp_path), dicts, hashes)["rows_out"] == 6
    args, new_hashes = feature_args(tmp_path), hashes
    if change in ("radius", "ngram", "dist_thres"):
        setattr(args, change, {"radius": 1, "ngram": 3, "dist_thres": 8.0}[change])
    elif change == "dictionary":
        new_hashes = cache_features.dictionary_hashes(changed_dictionaries(tmp_path))
    elif change == "resolver_version":
        monkeypatch.setattr(cache_features, "RESOLVER_VERSION", cache_features.RESOLVER_VERSION + 1)
    again = build_feature_split(manifest, out, args, dicts, new_hashes)
    assert (again.get("status") == "skipped_exists") == (change == "same")
    if change == "ngram":
        assert again["ngram"] == 3


def test_changed_dictionary_changes_item_keys(tmp_path):
    first = feature_utils.FeatureCache(tmp_path, "aa")
    second = feature_utils.FeatureCache(tmp_path, "bb")
    assert first.seq_key("MKV", 4) != second.seq_key("MKV", 4)
    assert first.smiles_key("CCO", 2) != second.smiles_key("CCO", 2)
    assert first.seq_key("MKV", 4) != first.seq_key("MKV", 3)
    assert first.smiles_key("CCO", 2) != first.smiles_key("CCO", 1)
