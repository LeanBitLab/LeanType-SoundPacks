#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from validate_pack import ValidationError, validate_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    if relative.startswith(".git/"):
        return True
    if path.name in {".DS_Store", "Thumbs.db"}:
        return True
    if path.suffix == ".pyc":
        return True
    return False


def package_pack(
    src_dir: Path,
    out_zip: Path,
    base_download_url: str | None = None
) -> None:
    src_dir = src_dir.resolve()
    out_zip = out_zip.resolve()

    manifest = validate_dir(src_dir)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue
            if should_skip(path, src_dir):
                continue
            relative = path.relative_to(src_dir).as_posix()
            zip_file.write(path, relative)

    checksum = sha256_file(out_zip)
    meta = {
        "id": manifest["id"],
        "name": manifest["name"],
        "summary": manifest.get("summary", ""),
        "author": manifest.get("author"),
        "license": manifest.get("license"),
        "versionCode": manifest["versionCode"],
        "versionName": manifest["versionName"],
        "minAppVersionCode": manifest.get("minAppVersionCode", 1),
        "tags": manifest.get("tags", []),
        "file": out_zip.name,
        "sha256": checksum,
        "sizeBytes": out_zip.stat().st_size,
    }

    if base_download_url:
        meta["downloadUrl"] = f"{base_download_url.rstrip('/')}/{out_zip.name}"

    if manifest.get("preview"):
        meta["previewFile"] = manifest["preview"]

    if manifest.get("icon"):
        meta["iconFile"] = manifest["icon"]

    meta_path = out_zip.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Created: {out_zip}")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Package a LeanType sound pack into a zip file."
    )
    parser.add_argument("source", help="Pack directory containing pack.json")
    parser.add_argument("output", help="Output zip file path")
    parser.add_argument("--base-url", help="Base download URL used in metadata")

    args = parser.parse_args()

    try:
        package_pack(
            Path(args.source),
            Path(args.output),
            args.base_url
        )
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
