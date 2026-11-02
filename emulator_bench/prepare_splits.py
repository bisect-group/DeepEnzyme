import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (  # noqa: E402
    DEFAULT_SPLIT_GROUPS,
    KEY_COLUMNS,
    atomic_json,
    discover_split_jobs,
    manifest_path,
    normalize_threshold_args,
    read_table,
    require_columns,
    stable_hash,
    write_table,
)
from emulator_bench.structures import RESOLVER_VERSION, SOURCES, StructureResolver, resolve_sequences, sequence_candidates  # noqa: E402

DEFAULT_STRUCTURE_ROOT = REPO_ROOT.parents[1] / "data/intermediate"
STRUCTURE_FOLDERS = (("mmcif_dir", "mmcif"), ("alphafold_dir", "alphafold"), ("esmfold2_dir", "esmfold2_pdb"), ("esm3_dir", "esm"))


def _sample_id(row: dict) -> str:
    return stable_hash(json.dumps({column: str(row.get(column, "")) for column in KEY_COLUMNS}, sort_keys=True))


def _source_fingerprint(source_path: Path, limit_rows) -> str:
    stat = Path(source_path).stat()
    return stable_hash(json.dumps([str(Path(source_path).resolve()), stat.st_size, stat.st_mtime_ns, limit_rows or 0]))


def prepared_manifest_is_valid(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = read_table(path)
    except Exception:
        return False
    return "fingerprint" in frame.columns and set(frame["fingerprint"]) == {fingerprint}


def split_tables(jobs, limit_rows) -> dict:
    """Source tables by (job index, partition)."""
    tables = {}
    for index, job in enumerate(jobs):
        for split in ("train", "val", "test"):
            path = Path(job[f"{split}_path"])
            frame = read_table(path)
            require_columns(frame, KEY_COLUMNS, path)
            tables[index, split] = frame.head(limit_rows).copy() if limit_rows else frame
    return tables


def prepare_manifest(frame: pd.DataFrame, structures: pd.DataFrame, source_path: Path, args) -> pd.DataFrame:
    """One row per source row, with the structure of its sequence. Rows without a structure stay."""
    out = frame.merge(structures.rename(columns={"source": "structure_source"}), on="sequence", how="left", validate="many_to_one")
    out["sample_id"] = [_sample_id(row) for row in frame.to_dict("records")]
    out["target"] = out[args.target_col].astype(float)
    out["fingerprint"] = stable_hash(
        json.dumps([RESOLVER_VERSION, _source_fingerprint(source_path, args.limit_rows), out["structure_path"].tolist()])
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepEnzyme EMULaToR split manifests with one structure per sequence.")
    parser.add_argument("--base_dir", type=Path, required=True, help="Task directory, e.g. data/processed/datasets/kinetic_dataset/kcat.")
    parser.add_argument("--manifests_dir", type=Path, required=True)
    parser.add_argument("--chains_dir", type=Path, required=True, help="Folder for the extracted experimental chains.")
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--structure_root", type=Path, default=DEFAULT_STRUCTURE_ROOT)
    for name, sub in STRUCTURE_FOLDERS:
        parser.add_argument(f"--{name}", type=Path, default=None, help=f"Default: <structure_root>/{sub}")
    parser.add_argument("--workers", type=int, default=8, help="Processes that extract chains from mmCIF files.")
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    jobs = discover_split_jobs(args.base_dir, args.split_groups, args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    dirs = [getattr(args, name) or args.structure_root / sub for name, sub in STRUCTURE_FOLDERS]
    resolver = StructureResolver.from_dirs(*dirs, chains_dir=args.chains_dir)
    tables = split_tables(jobs, args.limit_rows)
    sequences = sorted(set().union(*(set(frame["sequence"]) for frame in tables.values())))
    candidates = sequence_candidates(args.base_dir.parent, [args.base_dir.name], sequences)
    structures = resolve_sequences(resolver, candidates, sequences, args.workers)

    summaries = []
    for (index, split), frame in tables.items():
        job = jobs[index]
        out_path = manifest_path(args.manifests_dir, job, split)
        manifest = prepare_manifest(frame, structures, Path(job[f"{split}_path"]), args)
        if not args.overwrite and prepared_manifest_is_valid(out_path, manifest["fingerprint"].iloc[0]):
            status = "skipped_exists"
        else:
            write_table(out_path, manifest)
            status = "written"
        counts = manifest["structure_source"].value_counts()
        summaries.append(
            {
                "split_group": job["split_group"],
                "split_name": job["split_name"],
                "partition": split,
                "status": status,
                "n_rows": len(manifest),
                **{source: int(counts.get(source, 0)) for source in (*SOURCES, "none")},
            }
        )
    summary = {"resolver_version": RESOLVER_VERSION, "jobs": len(jobs), "splits": summaries}
    atomic_json(Path(args.manifests_dir) / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
