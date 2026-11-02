import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from rdkit import Chem
from sklearn.metrics import pairwise_distances

CACHE_VERSION = "v6"


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_signature(path):
    p = Path(path)
    st = p.stat()
    payload = {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    return json.dumps(payload, sort_keys=True)


def split_sequence(sequence, ngram, word_dict):
    sequence = "--" + str(sequence) + "="
    words = []
    for i in range(len(sequence) - ngram + 1):
        token = sequence[i : i + ngram]
        words.append(word_dict.get(token, 0))
    return np.array(words, dtype=np.int64)


def unknown_id(table):
    """Id of an atom or bond type that is not in the dictionary. It is not added to the dictionary."""
    return len(table)


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        idx = a.GetIdx()
        atoms[idx] = (atoms[idx], "aromatic")
    atoms = [atom_dict.get(a, unknown_id(atom_dict)) for a in atoms]
    return np.array(atoms, dtype=np.int64)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict.get(str(b.GetBondType()), unknown_id(bond_dict))
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    if len(i_jbond_dict) < mol.GetNumAtoms():
        # Atoms without a bond get an empty entry, and all entries follow the atom index.
        i_jbond_dict = {i: i_jbond_dict.get(i, []) for i in range(mol.GetNumAtoms())}
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    """The original WL step, with one fingerprint per atom.

    Where every atom has a bond, the result equals the original. Unseen single-atom fingerprints map to id 0,
    like the unseen fingerprints and edges of the original.
    """
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict.get(a, 0) for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict.get(fingerprint, 0))
            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                _i_jedge_dict[i]
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    _i_jedge_dict[i].append((j, edge_dict.get((both_side, edge), 0)))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints, dtype=np.int64)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=np.float32)


def chain_residues(pdb_path):
    """Residue letters and C-alpha coordinates of chain A, read by the fixed columns of the PDB format."""
    letters, coords = [], []
    with open(pdb_path) as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
                letters.append(seq1(line[17:20], custom_map={"MSE": "M"}, undef_code="X"))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return "".join(letters), np.asarray(coords).reshape(-1, 3)


def _make_aligner():
    # Global alignment with free end gaps: tags and missing termini cost nothing, inner gaps and mismatches do.
    aligner = PairwiseAligner(mode="global", match_score=2.0, mismatch_score=-1.0, open_gap_score=-4.0, extend_gap_score=-1.0)
    aligner.end_insertion_score = aligner.end_deletion_score = 0.0
    return aligner


ALIGNER = _make_aligner()


def align_chain(chain_seq, seq):
    """Dataset positions and chain indices of the aligned residue pairs, as two arrays."""
    if chain_seq == seq:
        index = np.arange(len(seq))
        return index, index
    blocks = ALIGNER.align(seq, chain_seq)[0].aligned
    positions = np.concatenate([np.arange(*block) for block in blocks[0]])
    chain_index = np.concatenate([np.arange(*block) for block in blocks[1]])
    return positions, chain_index


def protein_contact_map(pdb_path, seq, dist_thres=10.0):
    """Contact map over the dataset sequence as (csr matrix, aligned residues, aligned residues with the same letter).

    The chain sequence is aligned to the dataset sequence. Each aligned C-alpha sits at its dataset position. Positions without
    an aligned C-alpha have no contacts (a unit diagonal only), and chain residues without a partner are cut.
    """
    chain_seq, coords = chain_residues(pdb_path)
    positions, chain_index = align_chain(chain_seq, seq)
    n_match = sum(seq[p] == chain_seq[c] for p, c in zip(positions, chain_index))
    cont_arr = (pairwise_distances(coords[chain_index]) < dist_thres).astype(np.int32) if len(positions) else np.zeros((0, 0), np.int32)
    if len(positions) == len(seq):
        return sparse.csr_matrix(cont_arr), len(positions), n_match
    return padded_contact_map(cont_arr, len(seq), positions), len(positions), n_match


def padded_contact_map(cont_arr, target_len, positions=None):
    """Float64 map with a unit diagonal and contacts at the given positions (the first rows when None).

    The original builds and normalises its padded map in float64. Without a structure, cont_arr is empty and the model sees only self-loops.
    """
    positions = np.arange(cont_arr.shape[0]) if positions is None else positions
    padded = np.zeros((target_len, target_len), dtype=np.float64)
    padded[np.ix_(positions, positions)] = cont_arr
    np.fill_diagonal(padded, 1)
    return sparse.csr_matrix(padded)


class FeatureCache:
    def __init__(self, cache_dir, dict_hash):
        self.dict_hash = dict_hash
        self.cache_root = _ensure_dir(Path(cache_dir) / f"features_{CACHE_VERSION}")
        self.seq_dir = _ensure_dir(self.cache_root / "sequence")
        self.smiles_dir = _ensure_dir(self.cache_root / "smiles")
        self.struct_dir = _ensure_dir(self.cache_root / "structure")

    def _path(self, namespace_dir, key, suffix):
        return namespace_dir / key[:2] / f"{key}.{suffix}"

    def seq_key(self, sequence, ngram):
        payload = f"{CACHE_VERSION}|seq|dicts={self.dict_hash}|ngram={ngram}|{sequence}"
        return _sha256_text(payload)

    def smiles_key(self, smiles, radius):
        payload = f"{CACHE_VERSION}|smiles|dicts={self.dict_hash}|radius={radius}|{smiles}"
        return _sha256_text(payload)

    def struct_key(self, structure_path, sequence, dist_thres):
        sig = _file_signature(structure_path)
        payload = f"{CACHE_VERSION}|struct|dist={dist_thres}|seq={_sha256_text(sequence)}|{sig}"
        return _sha256_text(payload)

    def load_sequence(self, key):
        fpath = self._path(self.seq_dir, key, "npy")
        if not fpath.exists():
            return None
        try:
            return np.load(fpath, allow_pickle=False)
        except Exception:
            return None

    def save_sequence(self, key, arr):
        fpath = self._path(self.seq_dir, key, "npy")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npy")
        np.save(tmp, np.asarray(arr, dtype=np.int64), allow_pickle=False)
        os.replace(tmp, fpath)

    def load_smiles(self, key):
        fpath = self._path(self.smiles_dir, key, "npz")
        if not fpath.exists():
            return None, None
        try:
            data = np.load(fpath, allow_pickle=False)
            return data["fingerprints"], data["adjacency"]
        except Exception:
            return None, None

    def save_smiles(self, key, fingerprints, adjacency):
        fpath = self._path(self.smiles_dir, key, "npz")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npz")
        np.savez_compressed(
            tmp,
            fingerprints=np.asarray(fingerprints, dtype=np.int64),
            adjacency=np.asarray(adjacency, dtype=np.float32),
        )
        os.replace(tmp, fpath)

    def load_structure(self, key):
        """(csr matrix, aligned residues, matching aligned residues), or None."""
        fpath = self._path(self.struct_dir, key, "npz")
        if not fpath.exists():
            return None
        try:
            with np.load(fpath, allow_pickle=False) as data:
                matrix = sparse.csr_matrix((data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"]))
                return matrix, int(data["n_aligned"]), int(data["n_match"])
        except Exception:
            return None

    def save_structure(self, key, matrix, n_aligned, n_match):
        fpath = self._path(self.struct_dir, key, "npz")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npz")
        np.savez_compressed(tmp, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape, n_aligned=n_aligned, n_match=n_match)
        os.replace(tmp, fpath)


def atomic_save_object_array(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp, np.array(values, dtype=object), allow_pickle=True)
    os.replace(tmp, path)
