#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

MAX_AUDIO_FILE_BYTES = 500 * 1024
MAX_PREVIEW_FILE_BYTES = 500 * 1024
MAX_ICON_FILE_BYTES = 200 * 1024
MAX_TOTAL_UNPACKED_BYTES = 20 * 1024 * 1024
MAX_FILES = 200
MAX_SOUND_EVENTS = 32
MAX_VARIANTS_PER_EVENT = 16

ALLOWED_AUDIO_EXTENSIONS = {
    ".ogg",
    ".opus",
    ".wav",
    ".mp3"
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".webp",
    ".png"
}

ALLOWED_TEXT_EXTENSIONS = {
    ".txt",
    ".md"
}

ALLOWED_MANIFEST_EXTENSIONS = {
    ".json"
}

ALLOWED_EXTENSIONS = (
    ALLOWED_AUDIO_EXTENSIONS
    | ALLOWED_IMAGE_EXTENSIONS
    | ALLOWED_TEXT_EXTENSIONS
    | ALLOWED_MANIFEST_EXTENSIONS
)

ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class ValidationError(Exception):
    pass


def fail(message: str):
    raise ValidationError(message)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_resolve(base: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute():
        fail(f"Absolute path is not allowed: {relative}")
    if ".." in rel.parts:
        fail(f"Path traversal is not allowed: {relative}")
    full = (base / rel).resolve()
    if not is_relative_to(full, base):
        fail(f"Path escapes pack directory: {relative}")
    return full


def check_file_type(path: Path):
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        fail(f"File type not allowed: {path.name}")


def read_manifest(pack_dir: Path) -> dict:
    manifest_path = pack_dir / "pack.json"
    if not manifest_path.is_file():
        fail("pack.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"pack.json is not valid JSON: {error}")
    if not isinstance(manifest, dict):
        fail("pack.json must contain a JSON object")
    return manifest


def validate_manifest_fields(manifest: dict):
    schema_version = manifest.get("schemaVersion")
    if schema_version != 1:
        fail("Unsupported schemaVersion. Expected 1.")

    pack_id = manifest.get("id")
    if not isinstance(pack_id, str) or not pack_id:
        fail("id is required")

    if not ID_PATTERN.match(pack_id):
        fail("id may only contain letters, numbers, dots, dashes, and underscores")

    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        fail("name is required")

    version_code = manifest.get("versionCode")
    if not isinstance(version_code, int) or version_code <= 0:
        fail("versionCode must be a positive integer")

    version_name = manifest.get("versionName")
    if not isinstance(version_name, str) or not version_name.strip():
        fail("versionName is required")

    default_volume = manifest.get("defaultMasterVolume", 0.85)
    if not isinstance(default_volume, (int, float)):
        fail("defaultMasterVolume must be a number")

    if default_volume < 0 or default_volume > 1:
        fail("defaultMasterVolume must be between 0 and 1")

    sounds = manifest.get("sounds")
    if not isinstance(sounds, dict) or not sounds:
        fail("sounds must be a non-empty object")

    if len(sounds) > MAX_SOUND_EVENTS:
        fail(f"Too many sound events. Maximum is {MAX_SOUND_EVENTS}.")

    for event, sound in sounds.items():
        if not isinstance(sound, dict):
            fail(f"Sound entry for {event} must be an object")

        files = sound.get("files")
        if not isinstance(files, list) or not files:
            fail(f"Sound entry for {event} must contain files array")

        if len(files) > MAX_VARIANTS_PER_EVENT:
            fail(f"Too many variants for {event}. Maximum is {MAX_VARIANTS_PER_EVENT}.")

        mode = sound.get("mode", "single")
        if mode not in {"single", "random", "cycle"}:
            fail(f"Invalid mode for {event}: {mode}")

        volume = sound.get("volume", 1.0)
        if not isinstance(volume, (int, float)):
            fail(f"Volume for {event} must be a number")

        if volume < 0 or volume > 1:
            fail(f"Volume for {event} must be between 0 and 1")


def validate_file_sizes(pack_dir: Path, manifest: dict):
    total_size = 0
    file_count = 0

    for path in pack_dir.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(pack_dir).as_posix()
        if relative.startswith(".git/"):
            continue
        if path.name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.name.startswith("."):
            fail(f"Hidden files are not allowed: {relative}")

        file_count += 1
        if file_count > MAX_FILES:
            fail(f"Too many files. Maximum is {MAX_FILES}.")

        check_file_type(path)
        size = path.stat().st_size
        total_size += size
        suffix = path.suffix.lower()

        if suffix in ALLOWED_AUDIO_EXTENSIONS:
            if size > MAX_AUDIO_FILE_BYTES:
                fail(f"Audio file too large: {relative}")

        if relative == manifest.get("preview"):
            if size > MAX_PREVIEW_FILE_BYTES:
                fail("Preview file is too large")

        if relative == manifest.get("icon"):
            if size > MAX_ICON_FILE_BYTES:
                fail("Icon file is too large")

    if total_size > MAX_TOTAL_UNPACKED_BYTES:
        fail("Pack is too large after extraction")


def validate_referenced_files(pack_dir: Path, manifest: dict):
    sounds = manifest.get("sounds", {})
    for event, sound in sounds.items():
        for rel in sound.get("files", []):
            if not isinstance(rel, str):
                fail(f"File entry for {event} must be a string")
            full = safe_resolve(pack_dir, rel)
            if not full.is_file():
                fail(f"Missing audio file for {event}: {rel}")
            if full.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
                fail(f"Unsupported audio extension for {event}: {rel}")

    preview = manifest.get("preview")
    if preview:
        preview_path = safe_resolve(pack_dir, preview)
        if not preview_path.is_file():
            fail("Preview file is missing")

    icon = manifest.get("icon")
    if icon:
        icon_path = safe_resolve(pack_dir, icon)
        if not icon_path.is_file():
            fail("Icon file is missing")


def validate_dir(pack_dir: Path) -> dict:
    pack_dir = Path(pack_dir).resolve()
    if not pack_dir.is_dir():
        fail(f"Not a directory: {pack_dir}")
    manifest = read_manifest(pack_dir)
    validate_manifest_fields(manifest)
    validate_file_sizes(pack_dir, manifest)
    validate_referenced_files(pack_dir, manifest)
    return manifest


def main():
    if len(sys.argv) != 2:
        print("Usage: validate_pack.py <pack_directory>", file=sys.stderr)
        sys.exit(1)

    pack_dir = Path(sys.argv[1])
    try:
        validate_dir(pack_dir)
        print(f"OK: {pack_dir}")
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
