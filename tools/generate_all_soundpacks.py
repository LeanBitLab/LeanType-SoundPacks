#!/usr/bin/env python3
"""
LeanType Sound Pack Generator

Generates 10 algorithmically synthesized LeanType sound packs, packages them
into zip archives with pack.json, calculates SHA-256 checksums, and produces
a root index.json for direct raw repository download.

Requirements:
    pip install numpy scipy soundfile

Usage:
    python generate_all_soundpacks.py \
        --base-url https://raw.githubusercontent.com/YOUR_USER/leantype-soundpacks/main/dist

Notes:
    - Output format is OGG/Vorbis when supported by libsndfile.
    - If OGG is unavailable, output automatically falls back to WAV.
    - All sounds are mono, 48 kHz, short, peak-normalized to about -2 dBFS,
      and trimmed/faded to avoid leading silence and edge clicks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf


SAMPLE_RATE = 48000
TARGET_PEAK_DB = -2.0
MIN_DURATION = 0.025
MAX_DURATION = 0.150

MAX_AUDIO_FILE_BYTES = 500 * 1024
MAX_PACK_ZIP_BYTES = 2 * 1024 * 1024


# ----------------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------------

def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def clamp_freq(freq: float) -> float:
    nyq = SAMPLE_RATE / 2.0
    return float(max(20.0, min(freq, nyq - 100.0)))


def child_rng(rng: np.random.Generator, salt: int = 0) -> np.random.Generator:
    seed = int(rng.integers(0, 2**32 - 1))
    seed = (seed + salt * 2654435761) % (2**32)
    return np.random.default_rng(seed)


def finalize(
    audio: np.ndarray,
    target_db: float = TARGET_PEAK_DB,
    fade_in_ms: float = 0.4,
    fade_out_ms: float = 2.0,
) -> np.ndarray:
    """
    Normalize, remove DC, enforce max duration, and apply tiny fades.
    """
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    if audio.size == 0:
        return np.zeros(int(MIN_DURATION * SAMPLE_RATE), dtype=np.float32)

    max_samples = int(MAX_DURATION * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    audio = audio - float(np.mean(audio))

    n = len(audio)

    fade_in_samples = int(fade_in_ms * 0.001 * SAMPLE_RATE)
    fade_out_samples = int(fade_out_ms * 0.001 * SAMPLE_RATE)

    if fade_in_samples > 0:
        fade_in_samples = min(fade_in_samples, max(1, n // 2))
        audio[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples)

    if fade_out_samples > 0:
        fade_out_samples = min(fade_out_samples, max(1, n // 2))
        audio[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples)

    peak = float(np.max(np.abs(audio))) if n else 0.0
    if peak > 1e-9:
        audio *= db_to_linear(target_db) / peak

    return audio.astype(np.float32)


def noise(rng: np.random.Generator, duration: float) -> np.ndarray:
    n = max(1, int(round(duration * SAMPLE_RATE)))
    return rng.standard_normal(n)


def decay_env(duration: float, tau: float, attack_s: float = 0.0004) -> np.ndarray:
    n = max(1, int(round(duration * SAMPLE_RATE)))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    if tau <= 0:
        tau = max(duration / 6.0, 0.001)

    env = np.exp(-t / tau)

    attack_samples = max(1, int(attack_s * SAMPLE_RATE))
    if attack_samples > 1 and attack_samples < n:
        env[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)

    return env


def apply_sos(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if len(audio) < 16:
        return audio
    return sosfilt(sos, audio)


def lowpass(audio: np.ndarray, cutoff: float, order: int = 3) -> np.ndarray:
    cutoff = clamp_freq(cutoff)
    nyq = SAMPLE_RATE / 2.0

    if cutoff >= nyq - 100.0:
        return audio

    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return apply_sos(audio, sos)


def highpass(audio: np.ndarray, cutoff: float, order: int = 3) -> np.ndarray:
    cutoff = clamp_freq(cutoff)
    nyq = SAMPLE_RATE / 2.0

    if cutoff <= 20.0:
        return audio

    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return apply_sos(audio, sos)


def bandpass(audio: np.ndarray, low: float, high: float, order: int = 2) -> np.ndarray:
    nyq = SAMPLE_RATE / 2.0

    low = max(20.0, min(low, nyq - 100.0))
    high = max(20.0, min(high, nyq - 100.0))

    if high <= low:
        high = min(nyq - 100.0, low * 1.1)

    if high <= low:
        return audio

    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return apply_sos(audio, sos)


def mix(layers: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    max_len = max((len(a) for a, _ in layers if a is not None), default=0)
    out = np.zeros(max_len, dtype=np.float64)

    for audio, gain in layers:
        if audio is None or gain == 0.0:
            continue
        out[: len(audio)] += np.asarray(audio, dtype=np.float64) * float(gain)

    return out


def add_delayed(base: np.ndarray, layer: np.ndarray, delay_s: float, gain: float = 1.0) -> np.ndarray:
    delay = int(round(delay_s * SAMPLE_RATE))
    needed = max(len(base), delay + len(layer))

    out = np.zeros(needed, dtype=np.float64)
    out[: len(base)] += base
    out[delay : delay + len(layer)] += layer * float(gain)

    return out


def concat(arrays: List[np.ndarray], gap_s: float = 0.0) -> np.ndarray:
    if not arrays:
        return np.zeros(1, dtype=np.float64)

    gap_len = max(0, int(round(gap_s * SAMPLE_RATE)))
    gap = np.zeros(gap_len, dtype=np.float64)

    parts: List[np.ndarray] = []
    for i, array in enumerate(arrays):
        parts.append(array)
        if i != len(arrays) - 1 and gap_len > 0:
            parts.append(gap)

    return np.concatenate(parts)


def tone(
    duration: float,
    f0: float,
    f1: float | None = None,
    wave: str = "sine",
    curve: str = "exp",
    vibrato_hz: float = 0.0,
    vibrato_depth: float = 0.0,
) -> np.ndarray:
    n = max(1, int(round(duration * SAMPLE_RATE)))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    if f1 is None:
        freqs = np.full(n, float(f0), dtype=np.float64)
    else:
        if curve == "exp" and f0 > 0 and f1 > 0:
            freqs = float(f0) * (float(f1) / float(f0)) ** (t / max(duration, 1e-6))
        else:
            freqs = np.linspace(float(f0), float(f1), n)

    if vibrato_depth > 0 and vibrato_hz > 0:
        freqs = freqs * (1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_hz * t))

    phase = 2.0 * np.pi * np.cumsum(freqs) / SAMPLE_RATE

    if wave == "sine":
        y = np.sin(phase)
    elif wave == "square":
        y = np.sign(np.sin(phase))
    elif wave == "triangle":
        y = (2.0 / np.pi) * np.arcsin(np.sin(phase))
    elif wave == "saw":
        y = 2.0 * (phase / (2.0 * np.pi) - np.floor(0.5 + phase / (2.0 * np.pi)))
    else:
        y = np.sin(phase)

    return y


# ----------------------------------------------------------------------------
# Generic sound builders
# ----------------------------------------------------------------------------

def percussive(
    rng: np.random.Generator,
    duration: float,
    f0: float,
    f1: float | None = None,
    wave: str = "sine",
    tone_gain: float = 0.80,
    attack_dur: float = 0.006,
    attack_low: float | None = None,
    attack_high: float | None = None,
    attack_gain: float = 0.35,
    body_low: float | None = None,
    body_high: float | None = None,
    body_gain: float = 0.30,
    lowpass_cutoff: float = 4000.0,
    highpass_cutoff: float = 35.0,
    tau: float | None = None,
) -> np.ndarray:
    """
    Generic filtered percussive hit:
    tone body + transient noise attack + optional resonant noise body.
    """
    if f1 is None:
        f1 = f0 * 0.92

    if tau is None:
        tau = duration / 4.5

    body_tone = tone(duration, f0, f1, wave=wave) * decay_env(duration, tau)
    layers: List[Tuple[np.ndarray, float]] = [(body_tone, tone_gain)]

    if attack_gain > 0:
        attack = noise(rng, attack_dur)

        if attack_low and attack_high:
            attack = bandpass(attack, attack_low, attack_high)
        elif attack_high:
            attack = highpass(attack, attack_high)
        elif attack_low:
            attack = lowpass(attack, attack_low)

        attack *= decay_env(attack_dur, max(0.0015, attack_dur / 3.0))
        layers.append((attack, attack_gain))

    if body_gain > 0 and body_low and body_high:
        body_noise = noise(rng, duration)
        body_noise = bandpass(body_noise, body_low, body_high)
        body_noise *= decay_env(duration, duration / 5.0)
        layers.append((body_noise, body_gain))

    y = mix(layers)

    if highpass_cutoff > 0:
        y = highpass(y, highpass_cutoff)

    if lowpass_cutoff > 0:
        y = lowpass(y, lowpass_cutoff)

    return finalize(y)


def metallic(
    rng: np.random.Generator,
    duration: float,
    partials: Tuple[float, ...] = (2300.0, 3200.0, 4300.0, 5600.0),
    partial_gain: float = 0.16,
    attack_high: float = 1800.0,
    attack_gain: float = 0.50,
    body_low: float = 1800.0,
    body_high: float = 6500.0,
    body_gain: float = 0.25,
    thunk_freq: float = 90.0,
    thunk_gain: float = 0.35,
    lowpass_cutoff: float = 9000.0,
    highpass_cutoff: float = 80.0,
) -> np.ndarray:
    """
    Metallic click with detuned partials, noise burst, and optional low thunk.
    """
    layers: List[Tuple[np.ndarray, float]] = []

    for i, freq in enumerate(partials):
        detune = float(rng.uniform(0.99, 1.01))
        f = float(freq) * detune

        partial = tone(duration, f, f * 0.985, wave="sine")
        partial *= decay_env(duration, duration / (3.0 + i * 0.7))
        layers.append((partial, partial_gain * (0.9 ** i)))

    if attack_gain > 0:
        attack = noise(rng, 0.008)
        attack = highpass(attack, attack_high)
        attack *= decay_env(0.008, 0.003)
        layers.append((attack, attack_gain))

    if body_gain > 0 and body_low and body_high:
        body = noise(rng, duration)
        body = bandpass(body, body_low, body_high)
        body *= decay_env(duration, duration / 5.0)
        layers.append((body, body_gain))

    if thunk_freq and thunk_gain > 0:
        thunk = tone(duration * 0.6, thunk_freq * 1.2, thunk_freq * 0.8, wave="sine")
        thunk *= decay_env(duration * 0.6, duration / 6.0)
        layers.append((thunk, thunk_gain))

    y = mix(layers)

    if highpass_cutoff > 0:
        y = highpass(y, highpass_cutoff)

    if lowpass_cutoff > 0:
        y = lowpass(y, lowpass_cutoff)

    return finalize(y)


def chime(
    rng: np.random.Generator,
    duration: float,
    freqs: Tuple[float, ...] = (1318.0, 1976.0, 2637.0),
    gain: float = 0.30,
    attack_click: bool = True,
) -> np.ndarray:
    """
    Small mechanical/bell chime.
    """
    layers: List[Tuple[np.ndarray, float]] = []

    for i, freq in enumerate(freqs):
        partial = tone(duration, float(freq), float(freq) * 0.995, wave="sine")
        partial *= decay_env(duration, duration / (3.5 + i))
        layers.append((partial, gain * (0.8 ** i)))

    if attack_click:
        click = noise(rng, 0.006)
        click = highpass(click, 2500.0)
        click *= decay_env(0.006, 0.002)
        layers.append((click, 0.35))

    y = mix(layers)
    y = highpass(y, 120.0)
    y = lowpass(y, 12000.0)

    return finalize(y)


def chip_blip(
    duration: float,
    f0: float,
    f1: float | None = None,
    target_db: float = -3.0,
) -> np.ndarray:
    """
    Retro square-wave chiptune blip.
    """
    y = tone(duration, f0, f1, wave="square", curve="linear")
    y *= decay_env(duration, duration / 3.0, attack_s=0.0003)
    y = lowpass(y, 9500.0)
    return finalize(y, target_db=target_db)


def chip_sequence(
    notes: List[Tuple[float, float]],
    note_dur: float = 0.045,
    gap: float = 0.002,
) -> np.ndarray:
    parts = [chip_blip(note_dur, f0, f1) for f0, f1 in notes]
    return finalize(concat(parts, gap_s=gap))


def bubble(
    rng: np.random.Generator,
    duration: float,
    f0: float,
    f1: float,
    low: float = 300.0,
    high: float = 2600.0,
    pop_gain: float = 0.28,
) -> np.ndarray:
    """
    Liquid droplet / bubble pop.
    """
    y = tone(duration, f0, f1, wave="sine", curve="exp")
    y *= decay_env(duration, duration / 3.2)
    y = bandpass(y, low, high)

    pop_dur = min(0.009, duration * 0.25)
    pop = noise(rng, pop_dur)
    pop = lowpass(pop, 1800.0)
    pop *= decay_env(pop_dur, 0.003)

    y = mix([(y, 0.92), (pop, pop_gain)])
    y = highpass(y, 70.0)
    y = lowpass(y, 9000.0)

    return finalize(y)


def glass(
    rng: np.random.Generator,
    duration: float,
    base: float,
    partial_gain: float = 0.30,
    attack_gain: float = 0.16,
    body_gain: float = 0.12,
) -> np.ndarray:
    """
    Ceramic/glass marble tap.
    """
    partials = (
        base,
        base * 2.68,
        base * 4.05,
    )

    return metallic(
        rng,
        duration,
        partials=partials,
        partial_gain=partial_gain,
        attack_high=3200.0,
        attack_gain=attack_gain,
        body_low=base * 0.8,
        body_high=min(base * 3.5, 15000.0),
        body_gain=body_gain,
        thunk_freq=0.0,
        thunk_gain=0.0,
        lowpass_cutoff=13000.0,
        highpass_cutoff=180.0,
    )


def wood(
    rng: np.random.Generator,
    duration: float,
    base: float,
    brightness: float = 2.2,
    tone_gain: float = 0.72,
    res_gain: float = 0.42,
    attack_gain: float = 0.32,
    lowpass_cutoff: float = 8500.0,
    highpass_cutoff: float = 150.0,
) -> np.ndarray:
    """
    Acoustic woodblock mallet tap.
    """
    body = tone(duration, base * 1.07, base * 0.92, wave="sine")
    body *= decay_env(duration, duration / 5.5)

    resonance = noise(rng, duration)
    resonance = bandpass(
        resonance,
        base * 1.3,
        min(base * brightness, SAMPLE_RATE / 2.0 - 200.0),
    )
    resonance *= decay_env(duration, duration / 6.5)

    attack = noise(rng, 0.004)
    attack = highpass(attack, min(base * 2.0, SAMPLE_RATE / 2.0 - 200.0))
    attack *= decay_env(0.004, 0.0015)

    y = mix(
        [
            (body, tone_gain),
            (resonance, res_gain),
            (attack, attack_gain),
        ]
    )

    y = highpass(y, highpass_cutoff)
    y = lowpass(y, lowpass_cutoff)

    return finalize(y)


def ev(arrays: List[np.ndarray], mode: str | None = None, volume: float = 1.0) -> Dict:
    if mode is None:
        mode = "single" if len(arrays) == 1 else "random"

    return {
        "arrays": arrays,
        "mode": mode,
        "volume": float(volume),
    }


# ----------------------------------------------------------------------------
# Pack builders
# ----------------------------------------------------------------------------

def build_gateron_oil_king(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(68.0, 90.0))
        duration = float(r.uniform(0.056, 0.074))

        return percussive(
            r,
            duration,
            base * 1.35,
            base * 0.88,
            wave="sine",
            tone_gain=0.82,
            attack_dur=0.008,
            attack_low=1400.0,
            attack_gain=0.38,
            body_low=170.0,
            body_high=420.0,
            body_gain=0.32,
            lowpass_cutoff=3200.0,
            highpass_cutoff=36.0,
        )

    space = percussive(
        child_rng(rng, 20),
        0.110,
        70.0,
        46.0,
        wave="sine",
        tone_gain=0.85,
        attack_dur=0.010,
        attack_low=900.0,
        attack_gain=0.45,
        body_low=80.0,
        body_high=320.0,
        body_gain=0.35,
        lowpass_cutoff=2400.0,
        highpass_cutoff=30.0,
    )

    delete = percussive(
        child_rng(rng, 21),
        0.045,
        125.0,
        92.0,
        wave="sine",
        tone_gain=0.62,
        attack_dur=0.006,
        attack_high=800.0,
        attack_gain=0.35,
        body_low=180.0,
        body_high=500.0,
        body_gain=0.24,
        lowpass_cutoff=3000.0,
        highpass_cutoff=55.0,
    )

    ret = percussive(
        child_rng(rng, 22),
        0.095,
        82.0,
        58.0,
        wave="sine",
        tone_gain=0.80,
        attack_dur=0.008,
        attack_high=900.0,
        attack_gain=0.42,
        body_low=120.0,
        body_high=420.0,
        body_gain=0.30,
        lowpass_cutoff=2800.0,
        highpass_cutoff=35.0,
    )

    shift = percussive(
        child_rng(rng, 23),
        0.038,
        120.0,
        88.0,
        wave="sine",
        tone_gain=0.58,
        attack_dur=0.005,
        attack_low=1200.0,
        attack_gain=0.30,
        body_low=180.0,
        body_high=480.0,
        body_gain=0.20,
        lowpass_cutoff=3000.0,
        highpass_cutoff=60.0,
    )

    symbol = percussive(
        child_rng(rng, 24),
        0.052,
        96.0,
        66.0,
        wave="sine",
        tone_gain=0.70,
        attack_dur=0.007,
        attack_high=700.0,
        attack_gain=0.38,
        body_low=140.0,
        body_high=380.0,
        body_gain=0.28,
        lowpass_cutoff=2600.0,
        highpass_cutoff=40.0,
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_kailh_box_jade(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(4200.0, 5600.0))
        duration = float(r.uniform(0.036, 0.046))

        y = percussive(
            r,
            duration,
            base * 1.08,
            base * 0.96,
            wave="triangle",
            tone_gain=0.20,
            attack_dur=0.006,
            attack_high=2600.0,
            attack_gain=0.72,
            body_low=base * 0.8,
            body_high=min(base * 1.6, 14000.0),
            body_gain=0.48,
            lowpass_cutoff=9500.0,
            highpass_cutoff=1300.0,
        )

        click = noise(r, 0.005)
        click = highpass(click, 3000.0)
        click *= decay_env(0.005, 0.002)

        y = add_delayed(y, click, 0.010, 0.55)
        return finalize(y)

    def clicky_hit(seed: int, duration: float, base: float) -> np.ndarray:
        r = child_rng(rng, seed)

        y = percussive(
            r,
            duration,
            base * 1.06,
            base * 0.94,
            wave="triangle",
            tone_gain=0.18,
            attack_dur=0.006,
            attack_high=2400.0,
            attack_gain=0.72,
            body_low=base * 0.75,
            body_high=min(base * 1.6, 14000.0),
            body_gain=0.46,
            lowpass_cutoff=9000.0,
            highpass_cutoff=1100.0,
        )

        click = noise(r, 0.005)
        click = highpass(click, 2800.0)
        click *= decay_env(0.005, 0.002)

        y = add_delayed(y, click, 0.009, 0.5)
        return finalize(y)

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([clicky_hit(20, 0.062, 3200.0)], "single", 1.0),
        "keypress.delete": ev([clicky_hit(21, 0.034, 5200.0)], "single", 0.95),
        "keypress.return": ev([clicky_hit(22, 0.075, 3900.0)], "single", 0.95),
        "keypress.shift": ev([clicky_hit(23, 0.032, 5800.0)], "single", 0.90),
        "keypress.symbol": ev([clicky_hit(24, 0.048, 4700.0)], "single", 0.90),
    }


def build_holy_panda(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(210.0, 280.0))
        duration = float(r.uniform(0.052, 0.066))

        y = percussive(
            r,
            duration,
            base * 1.25,
            base * 0.82,
            wave="sine",
            tone_gain=0.76,
            attack_dur=0.007,
            attack_high=1300.0,
            attack_gain=0.50,
            body_low=180.0,
            body_high=480.0,
            body_gain=0.35,
            lowpass_cutoff=5200.0,
            highpass_cutoff=60.0,
        )

        bottom = tone(0.035, 115.0, 78.0, wave="sine")
        bottom *= decay_env(0.035, 0.012)

        y = add_delayed(y, bottom, 0.011, 0.55)
        return finalize(y)

    def tactile_hit(seed: int, duration: float, base: float, bottom_freq: float) -> np.ndarray:
        r = child_rng(rng, seed)

        y = percussive(
            r,
            duration,
            base * 1.22,
            base * 0.84,
            wave="sine",
            tone_gain=0.74,
            attack_dur=0.007,
            attack_high=1200.0,
            attack_gain=0.48,
            body_low=170.0,
            body_high=460.0,
            body_gain=0.33,
            lowpass_cutoff=5000.0,
            highpass_cutoff=55.0,
        )

        bottom = tone(0.035, bottom_freq * 1.1, bottom_freq * 0.8, wave="sine")
        bottom *= decay_env(0.035, 0.012)

        y = add_delayed(y, bottom, 0.012, 0.52)
        return finalize(y)

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([tactile_hit(20, 0.085, 150.0, 85.0)], "single", 1.0),
        "keypress.delete": ev([tactile_hit(21, 0.042, 320.0, 180.0)], "single", 0.95),
        "keypress.return": ev([tactile_hit(22, 0.090, 185.0, 95.0)], "single", 0.95),
        "keypress.shift": ev([tactile_hit(23, 0.036, 360.0, 210.0)], "single", 0.90),
        "keypress.symbol": ev([tactile_hit(24, 0.055, 260.0, 140.0)], "single", 0.90),
    }


def build_ibm_model_m(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        scale = float(r.uniform(0.92, 1.08))
        duration = float(r.uniform(0.105, 0.125))

        return metallic(
            r,
            duration,
            partials=(
                2300.0 * scale,
                3150.0 * scale,
                4200.0 * scale,
                5400.0 * scale,
            ),
            partial_gain=0.16,
            attack_high=1800.0,
            attack_gain=0.52,
            body_low=1600.0,
            body_high=7000.0,
            body_gain=0.22,
            thunk_freq=88.0 * scale,
            thunk_gain=0.38,
            lowpass_cutoff=9500.0,
            highpass_cutoff=85.0,
        )

    space = metallic(
        child_rng(rng, 20),
        0.130,
        partials=(2100.0, 2900.0, 3900.0, 5000.0),
        partial_gain=0.15,
        attack_high=1600.0,
        attack_gain=0.52,
        body_low=1400.0,
        body_high=6500.0,
        body_gain=0.22,
        thunk_freq=70.0,
        thunk_gain=0.45,
        lowpass_cutoff=9000.0,
        highpass_cutoff=70.0,
    )

    delete = metallic(
        child_rng(rng, 21),
        0.060,
        partials=(2600.0, 3600.0, 4700.0),
        partial_gain=0.15,
        attack_high=2000.0,
        attack_gain=0.52,
        body_low=1800.0,
        body_high=7500.0,
        body_gain=0.20,
        thunk_freq=110.0,
        thunk_gain=0.25,
        lowpass_cutoff=10000.0,
        highpass_cutoff=100.0,
    )

    ret = metallic(
        child_rng(rng, 22),
        0.140,
        partials=(2200.0, 3100.0, 4100.0, 5300.0),
        partial_gain=0.16,
        attack_high=1700.0,
        attack_gain=0.55,
        body_low=1500.0,
        body_high=6800.0,
        body_gain=0.22,
        thunk_freq=80.0,
        thunk_gain=0.42,
        lowpass_cutoff=9200.0,
        highpass_cutoff=75.0,
    )

    shift = metallic(
        child_rng(rng, 23),
        0.048,
        partials=(2900.0, 4000.0, 5200.0),
        partial_gain=0.14,
        attack_high=2200.0,
        attack_gain=0.50,
        body_low=2000.0,
        body_high=8000.0,
        body_gain=0.18,
        thunk_freq=0.0,
        thunk_gain=0.0,
        lowpass_cutoff=10500.0,
        highpass_cutoff=130.0,
    )

    symbol = metallic(
        child_rng(rng, 24),
        0.065,
        partials=(2500.0, 3400.0, 4600.0),
        partial_gain=0.15,
        attack_high=1900.0,
        attack_gain=0.50,
        body_low=1700.0,
        body_high=7200.0,
        body_gain=0.20,
        thunk_freq=140.0,
        thunk_gain=0.25,
        lowpass_cutoff=9800.0,
        highpass_cutoff=95.0,
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_royal_typewriter(rng: np.random.Generator) -> Dict[str, Dict]:
    def hammer(
        r: np.random.Generator,
        duration: float,
        scale: float,
        thunk: float,
    ) -> np.ndarray:
        return metallic(
            r,
            duration,
            partials=(
                3000.0 * scale,
                4700.0 * scale,
                6200.0 * scale,
            ),
            partial_gain=0.12,
            attack_high=2300.0,
            attack_gain=0.55,
            body_low=2000.0,
            body_high=7500.0,
            body_gain=0.18,
            thunk_freq=thunk,
            thunk_gain=0.18,
            lowpass_cutoff=11000.0,
            highpass_cutoff=250.0,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        return hammer(
            r,
            float(r.uniform(0.042, 0.054)),
            float(r.uniform(0.92, 1.08)),
            float(r.uniform(160.0, 210.0)),
        )

    def space_ratchet() -> np.ndarray:
        r = child_rng(rng, 20)
        ticks: List[np.ndarray] = []

        for i in range(3):
            tick = noise(r, 0.012)
            tick = bandpass(tick, 1500.0 + 220.0 * i, 3200.0 + 320.0 * i)
            tick *= decay_env(0.012, 0.004)

            thump = tone(0.018, 150.0 - 10.0 * i, 110.0 - 8.0 * i, wave="sine")
            thump *= decay_env(0.018, 0.006)

            ticks.append(
                finalize(
                    mix(
                        [
                            (tick, 0.72),
                            (thump, 0.38),
                        ]
                    ),
                    target_db=-4.0,
                )
            )

        return finalize(concat(ticks, gap_s=0.018))

    delete = hammer(child_rng(rng, 21), 0.042, 1.15, 220.0)

    ret = chime(
        child_rng(rng, 22),
        0.150,
        freqs=(1318.0, 1976.0, 2637.0),
        gain=0.30,
        attack_click=True,
    )

    clunk = tone(0.035, 160.0, 90.0, wave="sine")
    clunk *= decay_env(0.035, 0.012)
    ret = add_delayed(ret, clunk, 0.006, 0.35)
    ret = finalize(ret)

    shift = hammer(child_rng(rng, 23), 0.040, 1.30, 260.0)

    symbol = finalize(
        concat(
            [
                hammer(child_rng(rng, 24), 0.042, 1.05, 200.0),
                hammer(child_rng(rng, 25), 0.038, 1.20, 240.0),
            ],
            gap_s=0.014,
        )
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space_ratchet()], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_creamy_linear_jelly(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(88.0, 118.0))
        duration = float(r.uniform(0.055, 0.075))

        return percussive(
            r,
            duration,
            base * 1.18,
            base * 0.86,
            wave="sine",
            tone_gain=0.78,
            attack_dur=0.006,
            attack_low=800.0,
            attack_gain=0.16,
            body_low=110.0,
            body_high=320.0,
            body_gain=0.22,
            lowpass_cutoff=750.0,
            highpass_cutoff=28.0,
        )

    space = percussive(
        child_rng(rng, 20),
        0.105,
        66.0,
        48.0,
        wave="sine",
        tone_gain=0.82,
        attack_dur=0.009,
        attack_low=600.0,
        attack_gain=0.18,
        body_low=80.0,
        body_high=260.0,
        body_gain=0.25,
        lowpass_cutoff=600.0,
        highpass_cutoff=24.0,
    )

    delete = percussive(
        child_rng(rng, 21),
        0.046,
        128.0,
        94.0,
        wave="sine",
        tone_gain=0.60,
        attack_dur=0.005,
        attack_low=900.0,
        attack_gain=0.14,
        body_low=130.0,
        body_high=360.0,
        body_gain=0.18,
        lowpass_cutoff=900.0,
        highpass_cutoff=35.0,
    )

    ret = percussive(
        child_rng(rng, 22),
        0.088,
        84.0,
        60.0,
        wave="sine",
        tone_gain=0.76,
        attack_dur=0.007,
        attack_low=700.0,
        attack_gain=0.16,
        body_low=100.0,
        body_high=300.0,
        body_gain=0.22,
        lowpass_cutoff=700.0,
        highpass_cutoff=28.0,
    )

    shift = percussive(
        child_rng(rng, 23),
        0.038,
        138.0,
        100.0,
        wave="sine",
        tone_gain=0.55,
        attack_dur=0.004,
        attack_low=950.0,
        attack_gain=0.12,
        body_low=140.0,
        body_high=380.0,
        body_gain=0.16,
        lowpass_cutoff=950.0,
        highpass_cutoff=40.0,
    )

    symbol = percussive(
        child_rng(rng, 24),
        0.056,
        102.0,
        74.0,
        wave="sine",
        tone_gain=0.68,
        attack_dur=0.006,
        attack_low=800.0,
        attack_gain=0.14,
        body_low=110.0,
        body_high=320.0,
        body_gain=0.20,
        lowpass_cutoff=800.0,
        highpass_cutoff=30.0,
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_chiptune_arcade(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        f0 = float(r.choice([784.0, 880.0, 987.0, 1046.0]))
        return chip_blip(0.045, f0, f0 * 0.82)

    space = chip_blip(0.080, 300.0, 940.0)
    delete = chip_blip(0.070, 950.0, 220.0)

    ret = chip_sequence(
        [
            (659.0, 659.0),
            (880.0, 880.0),
            (1318.0, 1318.0),
        ],
        note_dur=0.042,
        gap=0.003,
    )

    shift = chip_blip(0.038, 1318.0, 1760.0)

    symbol = chip_sequence(
        [
            (987.0, 987.0),
            (1318.0, 1318.0),
        ],
        note_dur=0.036,
        gap=0.004,
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_ceramic_glass_marble(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(2100.0, 2700.0))
        duration = float(r.uniform(0.038, 0.048))
        return glass(r, duration, base)

    space = glass(child_rng(rng, 20), 0.070, 1200.0, partial_gain=0.32)
    delete = glass(child_rng(rng, 21), 0.035, 2600.0, partial_gain=0.28)

    ret = chime(
        child_rng(rng, 22),
        0.120,
        freqs=(1800.0, 2700.0, 3600.0),
        gain=0.28,
        attack_click=True,
    )

    shift = glass(child_rng(rng, 23), 0.035, 2900.0, partial_gain=0.27)

    symbol = finalize(
        concat(
            [
                glass(child_rng(rng, 24), 0.040, 2300.0),
                glass(child_rng(rng, 25), 0.040, 2800.0),
            ],
            gap_s=0.010,
        )
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_water_bubble_pop(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        f0 = float(r.uniform(950.0, 1400.0))
        f1 = f0 * float(r.uniform(0.32, 0.45))
        duration = float(r.uniform(0.045, 0.060))
        return bubble(r, duration, f0, f1, low=300.0, high=2600.0)

    space = bubble(child_rng(rng, 20), 0.100, 320.0, 110.0, low=120.0, high=900.0)
    delete = bubble(child_rng(rng, 21), 0.040, 1600.0, 700.0, low=400.0, high=3200.0)

    ret = finalize(
        concat(
            [
                bubble(child_rng(rng, 22), 0.055, 700.0, 350.0, low=200.0, high=1800.0),
                bubble(child_rng(rng, 23), 0.055, 1000.0, 450.0, low=250.0, high=2200.0),
            ],
            gap_s=0.008,
        )
    )

    shift = bubble(child_rng(rng, 24), 0.035, 1800.0, 900.0, low=500.0, high=3800.0)

    symbol = finalize(
        concat(
            [
                bubble(child_rng(rng, 25), 0.040, 1200.0, 600.0, low=300.0, high=2600.0),
                bubble(child_rng(rng, 26), 0.035, 1500.0, 800.0, low=350.0, high=3000.0),
            ],
            gap_s=0.007,
        )
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


def build_teak_woodblock(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(680.0, 860.0))
        duration = float(r.uniform(0.045, 0.060))
        return wood(r, duration, base)

    space = wood(
        child_rng(rng, 20),
        0.085,
        390.0,
        brightness=2.0,
        lowpass_cutoff=6000.0,
        highpass_cutoff=100.0,
    )

    delete = wood(
        child_rng(rng, 21),
        0.038,
        1500.0,
        brightness=2.6,
        lowpass_cutoff=9000.0,
        highpass_cutoff=350.0,
    )

    ret = finalize(
        concat(
            [
                wood(child_rng(rng, 22), 0.055, 720.0),
                wood(child_rng(rng, 23), 0.055, 560.0),
            ],
            gap_s=0.018,
        )
    )

    shift = wood(child_rng(rng, 24), 0.040, 1100.0)

    symbol = finalize(
        concat(
            [
                wood(child_rng(rng, 25), 0.042, 850.0),
                wood(child_rng(rng, 26), 0.036, 1250.0),
            ],
            gap_s=0.010,
        )
    )

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# Pack registry
# ----------------------------------------------------------------------------

@dataclass
class PackSpec:
    slug: str
    name: str
    summary: str
    tags: List[str]
    builder: Callable[[np.random.Generator], Dict[str, Dict]]
    master_volume: float = 0.85


PACKS: List[PackSpec] = [
    PackSpec(
        slug="gateron-oil-king-thock",
        name="Gateron Oil King Thock",
        summary="Deep, warm, lubricated mechanical switch with heavy spacebar clack.",
        tags=["mechanical", "thock", "linear", "deep"],
        builder=build_gateron_oil_king,
        master_volume=0.85,
    ),
    PackSpec(
        slug="kailh-box-jade-clicky",
        name="Kailh Box Jade Clicky",
        summary="High-pitch crisp tactile click-bar mechanism.",
        tags=["mechanical", "clicky", "tactile", "bright"],
        builder=build_kailh_box_jade,
        master_volume=0.80,
    ),
    PackSpec(
        slug="holy-panda-tactile",
        name="Holy Panda Tactile",
        summary="Snappy tactile bump with distinct bottom-out pop.",
        tags=["mechanical", "tactile", "snappy"],
        builder=build_holy_panda,
        master_volume=0.85,
    ),
    PackSpec(
        slug="ibm-model-m-beamspring",
        name="IBM Model M Beamspring",
        summary="Heavy vintage solenoid click with subtle metallic resonance.",
        tags=["mechanical", "buckling-spring", "vintage", "metallic"],
        builder=build_ibm_model_m,
        master_volume=0.82,
    ),
    PackSpec(
        slug="classic-1930s-royal-typewriter",
        name="Classic 1930s Royal Typewriter",
        summary="Metal hammer striker, ratchet spacebar, and carriage-return chime.",
        tags=["typewriter", "vintage", "mechanical", "chime"],
        builder=build_royal_typewriter,
        master_volume=0.82,
    ),
    PackSpec(
        slug="creamy-linear-jelly",
        name="Creamy Linear Jelly",
        summary="Muted, ultra-smooth dampened linear switch sound.",
        tags=["linear", "muted", "soft", "creamy"],
        builder=build_creamy_linear_jelly,
        master_volume=0.88,
    ),
    PackSpec(
        slug="8-bit-chiptune-arcade",
        name="8-Bit Chiptune Arcade",
        summary="Retro square-wave arcade blips and chirps.",
        tags=["retro", "chiptune", "8-bit", "arcade"],
        builder=build_chiptune_arcade,
        master_volume=0.72,
    ),
    PackSpec(
        slug="minimalistic-ceramic-glass-marble",
        name="Minimalistic Ceramic / Glass Marble",
        summary="Smooth polished mineral tap with glassy overtones.",
        tags=["minimal", "glass", "ceramic", "clean"],
        builder=build_ceramic_glass_marble,
        master_volume=0.80,
    ),
    PackSpec(
        slug="water-bubble-pop",
        name="Water Bubble Pop",
        summary="Resonant liquid droplet burst with soft pop.",
        tags=["liquid", "bubble", "soft", "playful"],
        builder=build_water_bubble_pop,
        master_volume=0.82,
    ),
    PackSpec(
        slug="acoustic-teak-woodblock",
        name="Acoustic Teak Woodblock",
        summary="Natural acoustic percussion mallet tap.",
        tags=["wood", "percussion", "acoustic", "warm"],
        builder=build_teak_woodblock,
        master_volume=0.85,
    ),
]


# ----------------------------------------------------------------------------
# Audio output and packaging
# ----------------------------------------------------------------------------

def supports_ogg_vorbis() -> bool:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp.close()

        sf.write(
            tmp.name,
            np.zeros(8, dtype=np.float32),
            SAMPLE_RATE,
            format="OGG",
            subtype="VORBIS",
        )

        Path(tmp.name).unlink(missing_ok=True)
        return True
    except Exception:
        return False


def write_audio(path: Path, audio: np.ndarray, use_ogg: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_ogg:
        sf.write(
            str(path),
            audio,
            SAMPLE_RATE,
            format="OGG",
            subtype="VORBIS",
        )
    else:
        sf.write(
            str(path),
            audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

    size = path.stat().st_size
    if size > MAX_AUDIO_FILE_BYTES:
        raise RuntimeError(
            f"Audio file too large: {path} ({size} bytes > {MAX_AUDIO_FILE_BYTES} bytes)"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def create_pack(
    spec: PackSpec,
    build_root: Path,
    use_ogg: bool,
) -> Tuple[Path, Dict, str]:
    seed = zlib.crc32(spec.slug.encode("utf-8"))
    rng = np.random.default_rng(seed)

    events = spec.builder(rng)

    pack_dir = build_root / spec.slug
    if pack_dir.exists():
        shutil.rmtree(pack_dir)

    audio_dir = pack_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ext = "ogg" if use_ogg else "wav"

    sounds: Dict[str, Dict] = {}
    preview_rel: str | None = None

    for event, data in events.items():
        arrays = data["arrays"]
        mode = data.get("mode")
        volume = float(data.get("volume", 1.0))

        if not arrays:
            continue

        files: List[str] = []

        for idx, audio in enumerate(arrays):
            if event == "keypress.default":
                stem = f"keypress_default_{idx + 1}"
            else:
                stem = event.split(".")[-1]
                if len(arrays) > 1:
                    stem = f"{stem}_{idx + 1}"

            rel = f"audio/{stem}.{ext}"
            write_audio(pack_dir / rel, audio, use_ogg)
            files.append(rel)

            if event == "keypress.default" and idx == 0:
                preview_rel = rel

        if mode is None:
            mode = "single" if len(files) == 1 else "random"

        sounds[event] = {
            "files": files,
            "mode": mode,
            "volume": volume,
        }

    if not sounds:
        raise RuntimeError(f"No sounds generated for pack: {spec.slug}")

    if preview_rel is None:
        first_event = next(iter(sounds.values()))
        preview_rel = first_event["files"][0]

    pack = {
        "schemaVersion": 1,
        "id": f"dev.leantype.sounds.{spec.slug}",
        "name": spec.name,
        "summary": spec.summary,
        "versionCode": 1,
        "versionName": "1.0.0",
        "author": "LeanType Sound Lab",
        "license": "CC0-1.0",
        "minAppVersionCode": 1,
        "defaultMasterVolume": spec.master_volume,
        "preview": preview_rel,
        "tags": spec.tags,
        "sounds": sounds,
    }

    pack_json_path = pack_dir / "pack.json"
    pack_json_path.write_text(
        json.dumps(pack, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return pack_dir, pack, preview_rel


def zip_pack(pack_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pack_dir.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(pack_dir).as_posix()
                zf.write(path, arcname)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LeanType sound packs with synthesized audio."
    )

    parser.add_argument(
        "--out",
        default="dist",
        help="Output directory for zips and index.json",
    )

    parser.add_argument(
        "--build",
        default="build",
        help="Temporary build directory",
    )

    parser.add_argument(
        "--base-url",
        default="https://raw.githubusercontent.com/YOUR_USER/leantype-soundpacks/main/dist",
        help=(
            "Base raw URL where dist files will be hosted. "
            "Example: https://raw.githubusercontent.com/user/repo/main/dist"
        ),
    )

    parser.add_argument(
        "--wav",
        action="store_true",
        help="Force WAV output even if OGG/Vorbis is available",
    )

    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep temporary build folder after packaging",
    )

    args = parser.parse_args()

    build_root = Path(args.build)
    dist_root = Path(args.out)

    if build_root.exists():
        shutil.rmtree(build_root)

    if dist_root.exists():
        shutil.rmtree(dist_root)

    build_root.mkdir(parents=True, exist_ok=True)
    dist_root.mkdir(parents=True, exist_ok=True)

    use_ogg = not args.wav and supports_ogg_vorbis()

    if not use_ogg:
        print("OGG/Vorbis output unavailable. Falling back to WAV.")

    index_packs: List[Dict] = []
    previews_dir = dist_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    for spec in PACKS:
        print(f"Generating: {spec.name}")

        pack_dir, pack, preview_rel = create_pack(spec, build_root, use_ogg)

        zip_name = f"{spec.slug}.zip"
        zip_path = dist_root / zip_name

        zip_pack(pack_dir, zip_path)

        zip_size = zip_path.stat().st_size
        if zip_size > MAX_PACK_ZIP_BYTES:
            raise RuntimeError(
                f"Pack zip too large: {zip_path} ({zip_size} bytes > {MAX_PACK_ZIP_BYTES} bytes)"
            )

        preview_src = pack_dir / preview_rel
        preview_dest = previews_dir / f"{spec.slug}_preview{preview_src.suffix}"
        shutil.copy2(preview_src, preview_dest)

        entry = {k: v for k, v in pack.items() if k != "sounds"}
        entry["file"] = zip_name
        entry["sha256"] = sha256_file(zip_path)
        entry["sizeBytes"] = zip_size
        entry["downloadUrl"] = f"{args.base_url.rstrip('/')}/{zip_name}"
        entry["previewUrl"] = f"{args.base_url.rstrip('/')}/previews/{preview_dest.name}"

        index_packs.append(entry)

        print(f"  -> {zip_path} ({zip_size} bytes)")

    index = {
        "schemaVersion": 1,
        "packs": index_packs,
    }

    index_path = dist_root / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.keep_build:
        shutil.rmtree(build_root)

    print(f"Done. Index written to: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
