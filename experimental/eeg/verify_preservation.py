from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_preparation.artifact_metadata import file_sha256
from experimental.eeg.runtime import PRESERVATION_ROOT


def verify_recovery_copy(root: Path) -> int:
    manifest_path = root / "preservation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    if len(files) != manifest["file_count"]:
        raise ValueError("Preservation inventory count differs.")
    for name, entry in files.items():
        recovered = root / name
        candidates = [recovered, Path(entry["source"])] if "source" in entry else [recovered]
        for path in candidates:
            if "symlink" in entry:
                if not path.is_symlink() or str(path.readlink()) != entry["symlink"]:
                    raise ValueError(f"Protected link changed: {path}")
            elif not path.is_file() or path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Protected file changed: {path}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recheck the frozen manuscript sources and recovery copy.")
    parser.add_argument("--preservation-root", type=Path, default=PRESERVATION_ROOT)
    args = parser.parse_args()
    print(f"Verified {verify_recovery_copy(args.preservation_root):,} protected inventory entries.")


if __name__ == "__main__":
    main()
