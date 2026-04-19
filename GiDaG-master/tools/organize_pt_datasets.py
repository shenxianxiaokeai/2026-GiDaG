import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def _normalized_stem(name: str) -> str:
    if name.endswith("_raw"):
        return name[: -len("_raw")]
    return name


def _target_from_source(src: Path, source_root: Path, target_root: Path) -> Path:
    rel = src.relative_to(source_root)
    stem = _normalized_stem(src.stem)

    # Keep nested groups (e.g., enron/*.pt) as a single dataset family folder.
    if len(rel.parts) > 1:
        group = rel.parts[0]
        return target_root / group / "{s}.pt".format(s=stem)

    # Time-series datasets: group by family before "_time".
    if "_time" in stem:
        family = stem.split("_time")[0]
        return target_root / family / "{s}.pt".format(s=stem)

    # Attribute-fused variants: group by dataset family before suffix.
    if stem.endswith("_attr_fused"):
        family = stem[: -len("_attr_fused")]
        return target_root / family / "{s}.pt".format(s=stem)

    # Default: one dataset per folder.
    return target_root / stem / "{s}.pt".format(s=stem)


def _ensure_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copy"


def organize_datasets(source_root: Path, target_root: Path) -> Dict:
    source_root = source_root.resolve()
    target_root = target_root.resolve()

    files = sorted(source_root.rglob("*.pt"))
    records: List[Dict] = []

    for src in files:
        dst = _target_from_source(src=src, source_root=source_root, target_root=target_root)
        action = _ensure_link_or_copy(src, dst)
        records.append(
            {
                "source": str(src),
                "target": str(dst),
                "action": action,
                "size_bytes": int(src.stat().st_size),
            }
        )

    manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "num_files": len(records),
        "records": records,
    }
    return manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "data" / "data"
    target_root = repo_root / "data"

    if not source_root.exists():
        raise FileNotFoundError("Source root not found: {p}".format(p=source_root))

    manifest = organize_datasets(source_root=source_root, target_root=target_root)
    manifest_path = target_root / "dataset_organization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Organized {n} files.".format(n=manifest["num_files"]))
    print("Manifest: {p}".format(p=manifest_path))


if __name__ == "__main__":
    main()
