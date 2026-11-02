import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_SPLIT_GROUPS,
    atomic_json,
    discover_split_jobs,
    feature_dir,
    load_json,
    manifest_path,
    normalize_threshold_args,
    read_table,
    stable_hash,
)
from emulator_bench.feature_utils import (
    CACHE_VERSION,
    FeatureCache,
    atomic_save_object_array,
    create_adjacency,
    create_atoms,
    create_ijbonddict,
    extract_fingerprints,
    load_pickle,
    padded_contact_map,
    protein_contact_map,
    split_sequence,
)
from emulator_bench.structures import RESOLVER_VERSION, SOURCES


REQUIRED_FEATURE_FILES = [
    "fingerprint.npy",
    "smileadjacencies.npy",
    "sequences.npy",
    "proteinadjacencies.npy",
    "logkcat.npy",
    "metadata.csv",
]

REQUIRED_MANIFEST_COLUMNS = {"sequence", "smiles", "value", "structure_path", "structure_source", "target"}
# Aligned residue pairs with a lower share of equal letters count as low identity. The row stays.
LOW_IDENTITY = 0.9
DICTIONARY_FILES = {
    "atom_dict": "atom_dict_0612.pickle",
    "bond_dict": "bond_dict_0612.pickle",
    "edge_dict": "edge_dict_0612.pickle",
    "fingerprint_dict": "fingerprint_dict_0612.pickle",
    "word_dict": "sequence_dict_0612.pickle",
}


LABEL = "log2_value"


def _signature(manifest: Path, limit_rows) -> str:
    stat = manifest.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}:{limit_rows}"


def dictionary_hashes(dict_dir: Path) -> dict:
    return {name: hashlib.sha256((Path(dict_dir) / name).read_bytes()).hexdigest() for name in DICTIONARY_FILES.values()}


def config_fingerprint(args, dict_hashes: dict) -> str:
    """Everything besides the manifest that changes the cached features."""
    payload = {
        "label": LABEL,
        "cache_version": CACHE_VERSION,
        "resolver_version": RESOLVER_VERSION,
        "radius": int(args.radius),
        "ngram": int(args.ngram),
        "dist_thres": float(args.dist_thres),
        "dictionaries": dict_hashes,
    }
    return stable_hash(json.dumps(payload, sort_keys=True))


def _feature_complete(out_dir: Path, manifest: Path, limit_rows, fingerprint: str) -> bool:
    """Complete features exist for this manifest and this configuration."""
    summary = load_json(out_dir / "feature_summary.json", {})
    return (
        all((out_dir / name).exists() for name in REQUIRED_FEATURE_FILES)
        and summary.get("config_fingerprint") == fingerprint
        and summary.get("manifest_signature") == _signature(manifest, limit_rows)
    )


def _label_for(row: dict) -> float:
    value = float(row["value"])
    if value <= 0:
        raise ValueError(f"value must be positive for the log2 label, got {value}")
    return float(np.log2(value))


def substrate_features(smiles: str, args, dictionaries: dict):
    """Fingerprint ids and adjacency of one molecule. Raises where the original code raises or the model cannot run."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    atoms = create_atoms(mol, dictionaries["atom_dict"])
    i_jbond = create_ijbonddict(mol, dictionaries["bond_dict"])
    fp_ids = extract_fingerprints(atoms, i_jbond, args.radius, dictionaries["fingerprint_dict"], dictionaries["edge_dict"])
    smiles_adj = create_adjacency(mol)
    if len(fp_ids) != len(smiles_adj):
        raise ValueError(f"Fingerprint length {len(fp_ids)} != atom count {len(smiles_adj)}: {smiles}")
    return fp_ids, smiles_adj


def build_feature_split(manifest: Path, out_dir: Path, args, dictionaries: dict, dict_hashes: dict) -> dict:
    fingerprint = config_fingerprint(args, dict_hashes)
    if _feature_complete(out_dir, manifest, args.limit_rows, fingerprint) and not args.overwrite:
        return {"manifest": str(manifest), "out_dir": str(out_dir), "status": "skipped_exists"}

    frame = read_table(manifest)
    if not REQUIRED_MANIFEST_COLUMNS.issubset(frame.columns):
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
        raise ValueError(f"Prepared manifest has an old or invalid schema; missing {missing} in {manifest}. Run prepare_splits.py first.")
    if args.limit_rows:
        frame = frame.head(args.limit_rows).copy()

    cache = FeatureCache(args.cache_dir, "".join(dict_hashes.values()))
    fingerprints, smileadjacencies, sequences, proteinadjacencies, labels = [], [], [], [], []
    kept_rows, failed_rows = [], []
    stats = {"sequence_hits": 0, "sequence_writes": 0, "smiles_hits": 0, "smiles_writes": 0, "structure_hits": 0, "structure_writes": 0}
    coverage, n_full, n_partial, n_low_identity = [], 0, 0, 0

    iterator = progress(frame.to_dict("records"), desc=f"Caching {manifest.parent.parent.name}/{manifest.parent.name}/{manifest.stem}", unit="row")
    for idx, row in enumerate(iterator):
        stage = "sequence"
        try:
            seq = str(row["sequence"])
            smiles = str(row["smiles"])
            structure_path = str(row["structure_path"])

            seq_key = cache.seq_key(seq, args.ngram)
            seq_ids = cache.load_sequence(seq_key)
            if seq_ids is None:
                seq_ids = split_sequence(seq, args.ngram, dictionaries["word_dict"])
                cache.save_sequence(seq_key, seq_ids)
                stats["sequence_writes"] += 1
            else:
                stats["sequence_hits"] += 1

            stage = "substrate"
            smiles_key = cache.smiles_key(smiles, args.radius)
            fp_ids, smiles_adj = cache.load_smiles(smiles_key)
            if fp_ids is None or smiles_adj is None:
                fp_ids, smiles_adj = substrate_features(smiles, args, dictionaries)
                cache.save_smiles(smiles_key, fp_ids, smiles_adj)
                stats["smiles_writes"] += 1
            else:
                stats["smiles_hits"] += 1

            stage = "protein"
            aligned = None
            if structure_path:
                struct_key = cache.struct_key(structure_path, seq, args.dist_thres)
                cached = cache.load_structure(struct_key)
                if cached is None:
                    cached = protein_contact_map(structure_path, seq, dist_thres=args.dist_thres)
                    cache.save_structure(struct_key, *cached)
                    stats["structure_writes"] += 1
                else:
                    stats["structure_hits"] += 1
                prot_adj, n_aligned, n_match = cached
                aligned = (n_aligned, n_match)
            else:
                prot_adj = padded_contact_map(np.zeros((0, 0)), len(seq))

            stage = "label"
            labels.append(np.asarray([_label_for(row)], dtype=np.float32))
            fingerprints.append(np.asarray(fp_ids, dtype=np.int64))
            smileadjacencies.append(np.asarray(smiles_adj, dtype=np.float32))
            sequences.append(np.asarray(seq_ids, dtype=np.int64))
            proteinadjacencies.append(prot_adj)
            kept_rows.append(idx)
            if aligned:
                n_aligned, n_match = aligned
                coverage.append(n_aligned / len(seq))
                n_full += n_aligned == len(seq)
                n_partial += n_aligned < len(seq)
                n_low_identity += n_aligned > 0 and n_match / n_aligned < LOW_IDENTITY
        except Exception as exc:
            failed_rows.append({"row_index": idx, "sample_id": row.get("sample_id", ""), "stage": stage, "error": str(exc)})
            if args.fail_on_error:
                raise

    if not kept_rows:
        raise RuntimeError(f"No valid rows were featurized for {manifest}")

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_object_array(out_dir / "fingerprint.npy", fingerprints)
    atomic_save_object_array(out_dir / "smileadjacencies.npy", smileadjacencies)
    atomic_save_object_array(out_dir / "sequences.npy", sequences)
    atomic_save_object_array(out_dir / "proteinadjacencies.npy", proteinadjacencies)
    atomic_save_object_array(out_dir / "logkcat.npy", labels)
    kept = frame.iloc[kept_rows]
    kept.to_csv(out_dir / "metadata.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(out_dir / "failed_rows.csv", index=False)

    sources = kept["structure_source"].value_counts()
    summary = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "rows_in": int(len(frame)),
        "rows_out": int(len(kept_rows)),
        "rows_failed": int(len(failed_rows)),
        "failed_by_stage": dict(Counter(item["stage"] for item in failed_rows)),
        "structure": {
            "n_rows": int(len(kept_rows)),
            "n_no_structure": int(sources.get("none", 0)),
            "n_aligned_full": int(n_full),
            "n_partial": int(n_partial),
            "mean_coverage": float(np.mean(coverage)) if coverage else None,
            "n_low_identity": int(n_low_identity),
            "sources": {source: int(sources.get(source, 0)) for source in (*SOURCES, "none")},
        },
        "label": LABEL,
        "cache_version": CACHE_VERSION,
        "config_fingerprint": fingerprint,
        "manifest_signature": _signature(manifest, args.limit_rows),
        "radius": int(args.radius),
        "ngram": int(args.ngram),
        "dist_thres": float(args.dist_thres),
        **stats,
    }
    atomic_json(out_dir / "feature_summary.json", summary)
    return summary


def load_dictionaries(dict_dir: Path):
    return {key: load_pickle(Path(dict_dir) / name) for key, name in DICTIONARY_FILES.items()}


def main():
    parser = argparse.ArgumentParser(description="Cache DeepEnzyme static features for prepared EMULaToR manifests.")
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--manifests_dir", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--dict_dir", type=Path, default=REPO_ROOT / "Data/Input")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--dist_thres", type=float, default=10.0)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    dictionaries = load_dictionaries(args.dict_dir)
    dict_hashes = dictionary_hashes(args.dict_dir)
    summaries = []
    for job in jobs:
        for split in ("train", "val", "test"):
            manifest = manifest_path(args.manifests_dir, job, split)
            if not manifest.exists():
                raise FileNotFoundError(f"Missing prepared manifest: {manifest}. Run prepare_splits.py first.")
            summaries.append(build_feature_split(manifest, feature_dir(args.features_dir, job, split), args, dictionaries, dict_hashes))

    payload = {"jobs": len(jobs), "summaries": summaries}
    atomic_json(args.features_dir / "cache_features_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
