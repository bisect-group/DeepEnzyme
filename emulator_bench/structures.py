"""One structure per protein sequence: the dataset's best experimental mmCIF chain, else AlphaFold, else ESMFold2, else ESM3.

This follows the structure rule of the CatPred baseline. Predicted structures are looked up by UniProt accession
(the text before "|"), ESMFold2 and ESM3 also by the full id with mutations. A sequence without a structure resolves to "none".
"""

import os
from multiprocessing import get_context
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

SEQUENCE_COL = "sequence"
# Bump when the structure rule changes, so manifests and features from the old rule are rebuilt.
RESOLVER_VERSION = 5
SOURCES = ("experimental", "alphafold", "esmfold2", "esm3")
CANDIDATE_COLUMNS = ["uniprot_id", "pdbs", "pdb_type", "pdb_chain", "pdb_coverage"]
ESM3_PREFIX, ESMFOLD2_PREFIX = "ESM3-open-small-", "ESMFold2-"


def norm_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "nat"} else text


def mmcif_index(names, directory: Path) -> dict:
    """PDB id to `<ID>_updated.cif`, else `<ID>.cif`."""
    index = {}
    for name in sorted(names):  # "X.cif" sorts before "X_updated.cif", which then wins
        if name.endswith(".cif"):
            index[name[:-4].removesuffix("_updated").upper()] = Path(directory) / name
    return index


def alphafold_index(names, directory: Path) -> dict:
    """Accession to the newest `AF-<accession>-F1-model_v<n>.pdb`."""
    best = {}
    for name in names:
        if name.startswith("AF-") and name.endswith(".pdb") and "-F1-model_v" in name:
            accession, _, version = name[3:-4].partition("-F1-model_v")
            if accession not in best or int(version) > best[accession][0]:
                best[accession] = (int(version), Path(directory) / name)
    return {accession: path for accession, (_, path) in best.items()}


def prefixed_index(names, directory: Path, prefix: str) -> dict:
    """Id to `<prefix><id>.pdb`."""
    return {name[len(prefix) : -4]: Path(directory) / name for name in names if name.startswith(prefix) and name.endswith(".pdb")}


def extract_chain(cif_path: Path, chain_id: str, out_path: Path) -> None:
    """Write the amino-acid atoms of one author chain as a PDB file that progres reads. Residues are numbered 1..n."""
    atoms = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True)
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain_id)]
    if atoms.array_length() == 0:
        raise ValueError(f"No amino-acid atoms for chain {chain_id} in {cif_path}")
    starts = np.zeros(atoms.array_length(), dtype=int)
    starts[struc.get_residue_starts(atoms)] = 1
    residue_index = np.cumsum(starts)
    lines = []
    for serial, (atom_name, res_name, number, (x, y, z)) in enumerate(zip(atoms.atom_name, atoms.res_name, residue_index, atoms.coord), start=1):
        name = atom_name if len(atom_name) == 4 else f" {atom_name:<3}"
        lines.append(f"ATOM  {serial % 100000:5d} {name} {res_name:>3} A{number % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
    lines.append("TER")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{out_path}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_path)


def _extract(job: tuple) -> None:
    extract_chain(*job)


class StructureResolver:
    """Resolves one structure per sequence from file indexes (id to path)."""

    def __init__(self, mmcif: dict, alphafold: dict, esmfold2: dict, esm3: dict, chains_dir: Path):
        self.mmcif, self.alphafold, self.esmfold2, self.esm3 = mmcif, alphafold, esmfold2, esm3
        self.chains_dir = Path(chains_dir)
        self.pending: dict = {}

    @classmethod
    def from_dirs(cls, mmcif_dir, alphafold_dir, esmfold2_dir, esm3_dir, chains_dir):
        def names(directory):
            return os.listdir(directory) if Path(directory).exists() else []

        return cls(mmcif_index(names(mmcif_dir), mmcif_dir), alphafold_index(names(alphafold_dir), alphafold_dir),
                   prefixed_index(names(esmfold2_dir), esmfold2_dir, ESMFOLD2_PREFIX),
                   prefixed_index(names(esm3_dir), esm3_dir, ESM3_PREFIX), chains_dir)

    def _experimental(self, candidates: list) -> dict | None:
        best, best_coverage = None, -1.0
        for item in candidates:
            pdb_id = norm_text(item.get("pdbs")).partition(":")[0].upper()
            chain = norm_text(item.get("pdb_chain"))
            coverage = float(item["pdb_coverage"]) if pd.notna(item.get("pdb_coverage")) else 0.0
            if norm_text(item.get("pdb_type")).lower() == "experimental" and chain and pdb_id in self.mmcif and coverage > best_coverage:
                best, best_coverage = (pdb_id, chain), coverage
        if best is None:
            return None
        pdb_id, chain = best
        out_path = self.chains_dir / f"{pdb_id}_{chain}.pdb"
        if not out_path.exists():
            self.pending[out_path] = (self.mmcif[pdb_id], chain, out_path)
        return {"source": "experimental", "structure_path": str(out_path), "structure_id": f"{pdb_id}_{chain}"}

    def resolve(self, uniprot_ids: list, candidates: list) -> dict:
        """candidates: dicts with pdbs ("ID:accession"), pdb_type, pdb_chain, pdb_coverage. uniprot_ids: ids of the sequence."""
        found = self._experimental(candidates)
        if found:
            return found
        ids = [norm_text(u) for u in uniprot_ids]
        accessions = [u.split("|", 1)[0].strip() for u in ids]
        for accession in accessions:
            if accession in self.alphafold:
                return {"source": "alphafold", "structure_path": str(self.alphafold[accession]), "structure_id": accession}
        for source, index in (("esmfold2", self.esmfold2), ("esm3", self.esm3)):
            for key in (k for pair in zip(ids, accessions) for k in pair):
                if key in index:
                    return {"source": source, "structure_path": str(index[key]), "structure_id": key}
        return {"source": "none", "structure_path": "", "structure_id": ""}

    def extract_pending(self, workers: int) -> None:
        jobs, self.pending = list(self.pending.values()), {}
        if jobs:
            with get_context("fork").Pool(min(workers, len(jobs))) as pool:
                for _ in pool.imap_unordered(_extract, jobs, chunksize=16):
                    pass


def sequence_candidates(data_root: Path, tasks, sequences) -> pd.DataFrame:
    """Candidate structure rows of the given sequences, from the `<task>_kinetic_params_3d.parquet` files, in a fixed order."""
    keep = set(sequences)
    frames = []
    for task in tasks:
        frame = pd.read_parquet(Path(data_root) / task / f"{task}_kinetic_params_3d.parquet", columns=[SEQUENCE_COL, *CANDIDATE_COLUMNS])
        frames.append(frame[frame[SEQUENCE_COL].isin(keep)])
    return pd.concat(frames).drop_duplicates().sort_values([SEQUENCE_COL, "pdbs", "pdb_chain"], na_position="last", kind="stable")


def resolve_sequences(resolver: StructureResolver, candidates: pd.DataFrame, sequences, workers: int = 8) -> pd.DataFrame:
    """One row per sequence: source, structure_path and structure_id. Experimental chains are extracted here."""
    groups = {seq: frame for seq, frame in candidates.groupby(SEQUENCE_COL, sort=False)}
    empty = candidates.iloc[:0]
    rows = []
    for seq in sequences:
        frame = groups.get(seq, empty)
        rows.append({SEQUENCE_COL: seq, **resolver.resolve(frame["uniprot_id"].dropna().unique().tolist(), frame.to_dict("records"))})
    resolver.extract_pending(workers)
    return pd.DataFrame(rows)
