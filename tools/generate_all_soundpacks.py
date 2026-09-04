#!/usr/bin/env python3
import json
import math
import os
import random
import struct
import subprocess
import sys
import wave
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
PACKS_DIR = BASE_DIR / "packs"
DIST_DIR = BASE_DIR / "dist"
SAMPLE_RATE = 44100


def generate_wav(filepath: Path, samples: list[float]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for sample in samples:
            clamped = max(-1.0, min(1.0, sample))
            val = int(clamped * 32767.0)
            frames.extend(struct.pack("<h", val))
        wav.writeframes(frames)


def wav_to_ogg(wav_path: Path, ogg_path: Path):
    ogg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "libvorbis", "-q:a", "4",
        str(ogg_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if wav_path.exists():
        wav_path.unlink()


def synthesize_thock(duration_ms: float = 65, pitch_hz: float = 140, decay_speed: float = 40.0) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        env = math.exp(-t * decay_speed)
        # Transient click + low resonant body
        transient = math.sin(2 * math.pi * (pitch_hz * 3.5) * t) * math.exp(-t * 220)
        body = math.sin(2 * math.pi * pitch_hz * t + 0.2 * math.sin(2 * math.pi * (pitch_hz * 0.5) * t))
        noise = (random.random() * 2.0 - 1.0) * math.exp(-t * 300) * 0.25
        val = (transient * 0.45 + body * 0.55 + noise) * env
        samples.append(val * 0.85)
    return samples


def synthesize_clicky(duration_ms: float = 45, pitch_hz: float = 2400) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 70.0)
        click = math.sin(2 * math.pi * pitch_hz * t) * math.exp(-t * 180.0)
        body = math.sin(2 * math.pi * (pitch_hz * 0.2) * t)
        noise = (random.random() * 2.0 - 1.0) * math.exp(-t * 150.0) * 0.3
        val = (click * 0.6 + body * 0.2 + noise) * env
        samples.append(val * 0.8)
    return samples


def synthesize_typewriter(duration_ms: float = 85, is_enter: bool = False, is_space: bool = False) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        env = math.exp(-t * (25.0 if is_enter else 35.0))
        hammer = math.sin(2 * math.pi * 1800 * t) * math.exp(-t * 300)
        clack = math.sin(2 * math.pi * (280 if is_space else 420) * t)
        noise = (random.random() * 2.0 - 1.0) * math.exp(-t * 120) * 0.4
        bell = 0.0
        if is_enter:
            bell = math.sin(2 * math.pi * 3200 * t) * math.exp(-t * 12.0) * 0.6
        val = (hammer * 0.4 + clack * 0.35 + noise + bell) * env
        samples.append(val * 0.85)
    return samples


def synthesize_8bit(duration_ms: float = 50, freq: float = 440, sweep: float = 0.0) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    phase = 0.0
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        cur_freq = max(50.0, freq + sweep * t)
        phase += 2 * math.pi * cur_freq / SAMPLE_RATE
        # Square wave
        sq = 1.0 if (phase % (2 * math.pi)) < math.pi else -1.0
        env = math.exp(-t * 30.0)
        samples.append(sq * env * 0.45)
    return samples


def synthesize_bubble(duration_ms: float = 70, freq_start: float = 400, freq_end: float = 900) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    phase = 0.0
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        progress = i / num_samples
        cur_freq = freq_start + (freq_end - freq_start) * (progress ** 2)
        phase += 2 * math.pi * cur_freq / SAMPLE_RATE
        env = math.sin(progress * math.pi) * math.exp(-progress * 2.5)
        samples.append(math.sin(phase) * env * 0.8)
    return samples


def synthesize_woodblock(duration_ms: float = 55, freq: float = 750) -> list[float]:
    num_samples = int(SAMPLE_RATE * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 60.0)
        wood = math.sin(2 * math.pi * freq * t) + 0.3 * math.sin(2 * math.pi * (freq * 1.58) * t)
        impact = (random.random() * 2.0 - 1.0) * math.exp(-t * 400) * 0.3
        val = (wood * 0.7 + impact) * env
        samples.append(val * 0.85)
    return samples


def build_pack(pack_id: str, name: str, summary: str, tags: list[str], audio_builder: callable):
    pack_dir = PACKS_DIR / pack_id
    audio_dir = pack_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_builder(audio_dir)

    manifest = {
        "schemaVersion": 1,
        "id": f"dev.leantype.sounds.{pack_id}",
        "name": name,
        "summary": summary,
        "versionCode": 1,
        "versionName": "1.0.0",
        "author": "LeanType Sound Lab",
        "license": "CC0-1.0",
        "minAppVersionCode": 1,
        "defaultMasterVolume": 0.85,
        "preview": "audio/keypress_default_1.ogg",
        "tags": tags,
        "sounds": {
            "keypress.default": {
                "files": [
                    "audio/keypress_default_1.ogg",
                    "audio/keypress_default_2.ogg"
                ],
                "mode": "random",
                "volume": 1.0
            },
            "keypress.space": {
                "files": ["audio/space.ogg"],
                "mode": "single",
                "volume": 1.0
            },
            "keypress.delete": {
                "files": ["audio/delete.ogg"],
                "mode": "single",
                "volume": 0.95
            },
            "keypress.return": {
                "files": ["audio/return.ogg"],
                "mode": "single",
                "volume": 0.95
            }
        }
    }

    (pack_dir / "pack.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (pack_dir / "license.txt").write_text("CC0 1.0 Universal - Public Domain Dedication\n", encoding="utf-8")
    print(f"Generated pack: {name} -> {pack_dir}")


def main():
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Mechanical Thock
    def make_thock(audio_dir):
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg") if False else None
        generate_wav(audio_dir / "k1.wav", synthesize_thock(65, 145, 42))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_thock(65, 138, 40))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_thock(90, 110, 28))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_thock(55, 160, 50))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_thock(75, 130, 35))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "mechanical_thock",
        "Mechanical Thock",
        "Deep, lubricated switch thock with heavy spacebar clack.",
        ["mechanical", "thock", "keyboard"],
        make_thock
    )

    # 2. Box Jade Clicky
    def make_clicky(audio_dir):
        generate_wav(audio_dir / "k1.wav", synthesize_clicky(45, 2400))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_clicky(45, 2300))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_clicky(60, 1800))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_clicky(40, 2600))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_clicky(55, 2100))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "box_jade_clicky",
        "Kailh Box Jade Clicky",
        "Ultra-crisp high-pitched tactile click bar switches.",
        ["mechanical", "clicky", "tactile"],
        make_clicky
    )

    # 3. Vintage Typewriter
    def make_typewriter(audio_dir):
        generate_wav(audio_dir / "k1.wav", synthesize_typewriter(80))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_typewriter(80))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_typewriter(95, is_space=True))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_typewriter(65))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_typewriter(140, is_enter=True))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "vintage_typewriter",
        "Vintage Royal Typewriter",
        "Classic cast-iron hammer strike with newline carriage chime on Enter.",
        ["retro", "typewriter", "vintage"],
        make_typewriter
    )

    # 4. 8-Bit Chiptune Arcade
    def make_8bit(audio_dir):
        generate_wav(audio_dir / "k1.wav", synthesize_8bit(45, 523.25))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_8bit(45, 659.25))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_8bit(70, 260.0, sweep=400.0))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_8bit(50, 440.0, sweep=-500.0))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_8bit(90, 523.25, sweep=600.0))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "arcade_8bit",
        "8-Bit Retro Arcade",
        "Nostalgic chiptune square-wave gaming blips and chirps.",
        ["gaming", "8bit", "retro"],
        make_8bit
    )

    # 5. Water Bubble Pop
    def make_bubble(audio_dir):
        generate_wav(audio_dir / "k1.wav", synthesize_bubble(65, 450, 950))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_bubble(65, 520, 1050))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_bubble(85, 300, 700))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_bubble(50, 600, 1100))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_bubble(80, 400, 1200))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "bubble_pop",
        "Water Bubble / Pop",
        "Satisfying soft liquid bubble pop feedback.",
        ["bubble", "pop", "soft"],
        make_bubble
    )

    # 6. Teak Woodblock
    def make_woodblock(audio_dir):
        generate_wav(audio_dir / "k1.wav", synthesize_woodblock(55, 780))
        wav_to_ogg(audio_dir / "k1.wav", audio_dir / "keypress_default_1.ogg")

        generate_wav(audio_dir / "k2.wav", synthesize_woodblock(55, 840))
        wav_to_ogg(audio_dir / "k2.wav", audio_dir / "keypress_default_2.ogg")

        generate_wav(audio_dir / "sp.wav", synthesize_woodblock(75, 520))
        wav_to_ogg(audio_dir / "sp.wav", audio_dir / "space.ogg")

        generate_wav(audio_dir / "del.wav", synthesize_woodblock(45, 920))
        wav_to_ogg(audio_dir / "del.wav", audio_dir / "delete.ogg")

        generate_wav(audio_dir / "ret.wav", synthesize_woodblock(65, 660))
        wav_to_ogg(audio_dir / "ret.wav", audio_dir / "return.ogg")

    build_pack(
        "woodblock_teak",
        "Teak Woodblock Minimal",
        "Natural acoustic wooden mallet resonance.",
        ["wood", "organic", "minimal"],
        make_woodblock
    )

    # Package all packs into dist/ and generate index.json
    print("\n--- Packaging all packs ---")
    from package_pack import package_pack
    from generate_index import generate_index

    base_download_url = "https://raw.githubusercontent.com/LeanBitLab/leantype-soundpacks/main/dist"
    for pack_folder in sorted(PACKS_DIR.iterdir()):
        if pack_folder.is_dir() and (pack_folder / "pack.json").exists():
            zip_target = DIST_DIR / f"{pack_folder.name}.zip"
            package_pack(pack_folder, zip_target, base_download_url=base_download_url)

    generate_index(DIST_DIR, base_download_url, BASE_DIR / "index.json")
    print("\nAll sound packs generated and packaged successfully!")


if __name__ == "__main__":
    main()
