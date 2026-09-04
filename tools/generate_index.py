#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def generate_index(
    dist_dir: Path,
    base_url: str,
    output_path: Path
) -> None:
    packs = []

    for meta_path in sorted(dist_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("downloadUrl"):
            file_name = meta.get("file")
            if not file_name:
                raise Exception(f"Missing file name in {meta_path}")
            meta["downloadUrl"] = f"{base_url.rstrip('/')}/{file_name}"
        packs.append(meta)

    index = {
        "schemaVersion": 1,
        "packs": packs
    }

    output_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Generated index with {len(packs)} pack(s): {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LeanType sound pack index.json"
    )
    parser.add_argument("dist", help="Directory containing .zip and .meta.json files")
    parser.add_argument(
        "--base-url",
        default="https://raw.githubusercontent.com/LeanBitLab/leantype-soundpacks/main/dist",
        help="Base URL where zip files are published"
    )
    parser.add_argument(
        "--out",
        default="index.json",
        help="Output index file path"
    )

    args = parser.parse_args()
    generate_index(
        Path(args.dist),
        args.base_url,
        Path(args.out)
    )


if __name__ == "__main__":
    main()
