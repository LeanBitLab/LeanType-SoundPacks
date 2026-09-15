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


def ev(arrays: List[np.ndarray], mode: str | None = None, volume: float = 1.0) -> Dict:
    if mode is None:
        mode = "single" if len(arrays) == 1 else "random"
    return {
        "arrays": arrays,
        "mode": mode,
        "volume": float(volume),
    }


def chip_blip(
    duration: float,
    f0: float,
    f1: float | None = None,
    target_db: float = -3.0,
) -> np.ndarray:
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

def build_chiptune_arcade(rng: np.random.Generator) -> Dict[str, Dict]:
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        f0 = float(r.choice([784.0, 880.0, 987.0, 1046.0]))
        return chip_blip(0.045, f0, f0 * 0.82)

    space = chip_blip(0.080, 300.0, 940.0)
    delete = chip_blip(0.070, 950.0, 220.0)
    ret = chip_sequence(
        [(659.0, 659.0), (880.0, 880.0), (1318.0, 1318.0)],
        note_dur=0.042,
        gap=0.003,
    )
    shift = chip_blip(0.038, 1318.0, 1760.0)
    symbol = chip_sequence(
        [(987.0, 987.0), (1318.0, 1318.0)],
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
    def glass_hit(r, duration, base):
        return _ac_modal(
            duration,
            base,
            ratios=[1.0, 2.68, 4.05, 5.82],
            gains=[0.8, 0.35, 0.22, 0.12],
            taus=[duration / 3.0, duration / 4.0, duration / 5.0, duration / 6.0],
            rng=r,
        )
    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)
        base = float(r.uniform(2100.0, 2700.0))
        return finalize(glass_hit(r, float(r.uniform(0.038, 0.048)), base))

    space = finalize(glass_hit(child_rng(rng, 20), 0.070, 1200.0))
    delete = finalize(glass_hit(child_rng(rng, 21), 0.035, 2600.0))
    ret = finalize(glass_hit(child_rng(rng, 22), 0.090, 1800.0))
    shift = finalize(glass_hit(child_rng(rng, 23), 0.035, 2900.0))
    symbol = finalize(concat([glass_hit(child_rng(rng, 24), 0.040, 2300.0), glass_hit(child_rng(rng, 25), 0.040, 2800.0)], gap_s=0.010))
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
        return bubble(r, float(r.uniform(0.045, 0.060)), f0, f1)
    space = bubble(child_rng(rng, 20), 0.100, 320.0, 110.0, low=120.0, high=900.0)
    delete = bubble(child_rng(rng, 21), 0.040, 1600.0, 700.0, low=400.0, high=3200.0)
    ret = finalize(concat([bubble(child_rng(rng, 22), 0.055, 700.0, 350.0), bubble(child_rng(rng, 23), 0.055, 1000.0, 450.0)], gap_s=0.008))
    shift = bubble(child_rng(rng, 24), 0.035, 1800.0, 900.0)
    symbol = finalize(concat([bubble(child_rng(rng, 25), 0.040, 1200.0, 600.0), bubble(child_rng(rng, 26), 0.035, 1500.0, 800.0)], gap_s=0.007))
    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# LeanType Ultra-Realistic Procedural Sound Pack Engine
# Drop-in physical-modeling helpers + upgraded builder functions
# ----------------------------------------------------------------------------
#
# This file assumes the following primitives already exist in
# tools/generate_all_soundpacks.py:
#
#   SAMPLE_RATE
#   finalize(...)
#   tone(...)
#   decay_env(...)
#   noise(...)
#   lowpass(...)
#   highpass(...)
#   bandpass(...)
#   mix(...)
#   add_delayed(...)
#   concat(...)
#   ev(...)
#   child_rng(...)
#
# All new helpers are prefixed with _ac_ to avoid collisions.
# ----------------------------------------------------------------------------


def _ac_final(audio: np.ndarray) -> np.ndarray:
    """
    Compatibility wrapper for final normalization.
    """
    try:
        return finalize(
            audio,
            target_db=-2.0,
            fade_in_ms=0.3,
            fade_out_ms=2.0,
        )
    except TypeError:
        return finalize(audio, target_db=-2.0)


def _ac_brown_noise(rng: np.random.Generator, duration: float) -> np.ndarray:
    """
    Brownian noise: integrated white noise.
    Much more organic than raw white noise for friction, air movement,
    wooden bodies, drum shells, and cavity resonance excitation.
    """
    n = max(1, int(round(duration * SAMPLE_RATE)))
    x = np.cumsum(rng.standard_normal(n))
    x -= float(np.mean(x))

    peak = float(np.max(np.abs(x)))
    if peak > 1e-9:
        x /= peak

    return x


def _ac_softsat(audio: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """
    Gentle tanh soft saturation.

    Used to turn mathematically clean waveforms into non-linear physical
    collision material. This adds dense, organic harmonics without the
    brittle character of bare sine waves.
    """
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0 or drive <= 0.0:
        return audio

    y = np.tanh(drive * audio)

    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y /= peak

    return y


def _ac_delayed_decay(
    duration: float,
    delay_s: float,
    tau: float,
    attack_s: float = 0.0002,
) -> np.ndarray:
    """
    Creates an exponential decay envelope that starts after delay_s.

    This is used for multi-stage physical events:
      0.0 ms  : initial impact
      2.0 ms  : body displacement
      4.0 ms+ : cavity / resonant decay
    """
    n = max(1, int(round(duration * SAMPLE_RATE)))
    env = np.zeros(n, dtype=np.float64)

    start = int(round(delay_s * SAMPLE_RATE))
    if start >= n:
        return env

    seg_dur = (n - start) / SAMPLE_RATE
    seg = decay_env(seg_dur, max(0.001, tau), attack_s=attack_s)

    if len(seg) > n - start:
        seg = seg[: n - start]

    env[start:] = seg
    return env


def _ac_fm_impact(
    rng: np.random.Generator,
    duration: float,
    f0: float,
    f1: float,
    mod_ratio: float = 1.8,
    index0: float = 6.0,
    drive: float = 2.5,
) -> np.ndarray:
    """
    Non-linear collision impulse using fast exponential FM pitch sweep.

    Bare sine waves are avoided. Instead:
      - exponentially falling/rising carrier frequency
      - decaying FM index
      - small noise injection
      - tanh saturation

    This creates a dense physical contact transient.
    """
    n = max(1, int(round(duration * SAMPLE_RATE)))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    f0 = max(20.0, float(f0))
    f1 = max(20.0, float(f1))

    ratio = max(f1 / f0, 1e-6)
    freqs = f0 * (ratio ** (t / max(duration, 1e-5)))

    carrier_phase = 2.0 * np.pi * np.cumsum(freqs) / SAMPLE_RATE

    mod_freqs = freqs * float(mod_ratio)
    mod_phase = 2.0 * np.pi * np.cumsum(mod_freqs) / SAMPLE_RATE

    mod_env = np.exp(-t / max(duration / 6.0, 0.001))

    y = np.sin(carrier_phase + float(index0) * mod_env * np.sin(mod_phase))

    # Tiny stochastic component prevents sterile mathematical repetition.
    y += 0.10 * rng.standard_normal(n)

    return _ac_softsat(y, drive)


def _ac_noise_burst(
    rng: np.random.Generator,
    duration: float,
    low: float | None = None,
    high: float | None = None,
    drive: float = 2.0,
) -> np.ndarray:
    """
    Coloured friction noise burst.

    Uses brownian noise blended with a small amount of white noise, then
    filters and soft-saturates it. This avoids harsh synthetic white-noise
    clicks while still preserving physical contact detail.
    """
    brown = _ac_brown_noise(rng, duration)
    white = noise(rng, duration)

    x = 0.72 * brown + 0.28 * white
    x = lowpass(x, min(14000.0, SAMPLE_RATE / 2.0 - 500.0))

    nyq_limit = SAMPLE_RATE / 2.0 - 200.0

    if low is not None and high is not None:
        low = max(30.0, float(low))
        high = min(float(high), nyq_limit)

        if high <= low:
            high = min(nyq_limit, low * 1.1)

        if high > low:
            x = bandpass(x, low, high, order=2)

    elif high is not None:
        high = min(float(high), nyq_limit)
        x = highpass(x, high)

    elif low is not None:
        low = max(30.0, float(low))
        x = highpass(x, low)

    x *= decay_env(duration, max(0.0015, duration / 5.0), attack_s=0.0002)

    return _ac_softsat(x, drive)


def _ac_cavity(
    rng: np.random.Generator,
    duration: float,
    bands: List[Tuple[float, float, float]],
    drive: float = 1.4,
) -> np.ndarray:
    """
    Hollow-body / Helmholtz cavity model.

    Each band is:
        (center_frequency, bandwidth, gain)

    The cavity is excited by coloured noise and rendered with parallel
    resonant bandpass filters plus weak sine resonators.
    """
    n = max(1, int(round(duration * SAMPLE_RATE)))
    out = np.zeros(n, dtype=np.float64)

    src = _ac_noise_burst(rng, duration, drive=1.2)

    for freq, bw, gain in bands:
        freq = float(freq) * float(rng.uniform(0.980, 1.020))
        bw = max(15.0, float(bw))

        lo = max(30.0, freq - bw * 0.5)
        hi = min(SAMPLE_RATE / 2.0 - 200.0, freq + bw * 0.5)

        if hi <= lo:
            continue

        bp = bandpass(src, lo, hi, order=2)

        ring = tone(duration, freq, freq * 0.992, wave="sine", curve="exp")
        ring *= decay_env(duration, max(0.006, duration / 5.0), attack_s=0.0003)

        out += bp * float(gain)
        out += ring * float(gain) * 0.22

    return _ac_softsat(out, drive)


def _ac_modal(
    duration: float,
    base_freq: float,
    ratios: List[float],
    gains: List[float],
    taus: float | List[float],
    rng: np.random.Generator | None = None,
    detune: float = 0.004,
) -> np.ndarray:
    """
    Multi-modal resonance builder.

    Used for membranes, tines, bells, strings, wooden bodies, and metal parts.
    """
    if isinstance(taus, (int, float)):
        taus = [float(taus)] * len(ratios)
    else:
        taus = [float(t) for t in taus]

    while len(taus) < len(ratios):
        taus.append(taus[-1])

    layers: List[Tuple[np.ndarray, float]] = []

    for ratio, gain, tau in zip(ratios, gains, taus):
        freq = float(base_freq) * float(ratio)

        if rng is not None and detune > 0.0:
            freq *= float(rng.uniform(1.0 - detune, 1.0 + detune))

        partial = tone(duration, freq, freq * 0.995, wave="sine", curve="exp")
        partial *= decay_env(duration, max(0.001, tau), attack_s=0.00025)

        layers.append((partial, float(gain)))

    return mix(layers)


def _ac_switch_hit(
    rng: np.random.Generator,
    duration: float,
    bottom: float = 150.0,
    cavity_bands: List[Tuple[float, float, float]] | None = None,
    click_low: float = 3800.0,
    click_high: float = 5500.0,
    click_gain: float = 0.45,
    bottom_gain: float = 0.85,
    spring_freq: float = 1800.0,
    spring_gain: float = 0.07,
    drive: float = 2.6,
    lowpass_cutoff: float = 9000.0,
    highpass_cutoff: float = 45.0,
    second_click_delay: float = 0.0,
    second_click_gain: float = 0.0,
) -> np.ndarray:
    """
    Multi-stage mechanical switch physics.

    Stage 1: 0-2 ms high-frequency impact / click-bar snap.
    Stage 2: 2-12 ms low-mid body displacement / bottom-out.
    Stage 3: 4+ ms cavity decay and subtle spring/metal ring.
    """
    if cavity_bands is None:
        cavity_bands = [
            (260.0, 90.0, 0.65),
            (460.0, 130.0, 0.42),
        ]

    layers: List[Tuple[np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Stage 1: primary impact transient
    # ------------------------------------------------------------------
    click_fm = _ac_fm_impact(
        rng,
        duration,
        click_high * 0.95,
        click_low * 0.75,
        mod_ratio=2.1,
        index0=5.5,
        drive=drive * 0.75,
    )

    click_noise = _ac_noise_burst(
        rng,
        duration,
        low=click_low,
        high=click_high,
        drive=drive * 0.65,
    )

    click_env = _ac_delayed_decay(duration, 0.0, 0.0032, attack_s=0.00015)

    click = mix(
        [
            (click_fm, 0.75),
            (click_noise, click_gain),
        ]
    ) * click_env

    layers.append((click, 1.0))

    # ------------------------------------------------------------------
    # Optional secondary click: tactile snap, double click, housing impact
    # ------------------------------------------------------------------
    if second_click_gain > 0.0:
        r2 = child_rng(rng, 91)

        click2_fm = _ac_fm_impact(
            r2,
            duration,
            click_high * 0.82,
            click_low * 0.68,
            mod_ratio=2.0,
            index0=4.6,
            drive=drive * 0.70,
        )

        click2_noise = _ac_noise_burst(
            r2,
            duration,
            low=click_low * 0.90,
            high=click_high * 0.90,
            drive=drive * 0.60,
        )

        click2_env = _ac_delayed_decay(
            duration,
            second_click_delay,
            0.0036,
            attack_s=0.00015,
        )

        click2 = mix(
            [
                (click2_fm, 0.68),
                (click2_noise, click_gain * 0.90),
            ]
        ) * click2_env * float(second_click_gain)

        layers.append((click2, 1.0))

    # ------------------------------------------------------------------
    # Stage 2: body displacement / bottom-out impact
    # ------------------------------------------------------------------
    bottom_fm = _ac_fm_impact(
        rng,
        duration,
        bottom * 2.4,
        bottom * 0.82,
        mod_ratio=1.35,
        index0=4.8,
        drive=drive,
    )

    bottom_noise = _ac_noise_burst(
        rng,
        duration,
        low=bottom * 0.8,
        high=bottom * 4.0,
        drive=drive * 0.55,
    )

    bottom_env = _ac_delayed_decay(
        duration,
        0.002,
        max(0.006, duration / 7.5),
        attack_s=0.0002,
    )

    bottom = mix(
        [
            (bottom_fm, 0.80),
            (bottom_noise, 0.40),
        ]
    ) * bottom_env * float(bottom_gain)

    layers.append((bottom, 1.0))

    # ------------------------------------------------------------------
    # Stage 3: cavity decay
    # ------------------------------------------------------------------
    cavity = _ac_cavity(rng, duration, cavity_bands, drive=drive * 0.55)
    cavity_env = _ac_delayed_decay(
        duration,
        0.004,
        max(0.014, duration / 3.8),
        attack_s=0.0003,
    )

    layers.append((cavity * cavity_env, 0.90))

    # ------------------------------------------------------------------
    # Subtle spring / metallic after-ring
    # ------------------------------------------------------------------
    if spring_gain > 0.0:
        spring_f = float(spring_freq) * float(rng.uniform(0.980, 1.020))

        spring = tone(duration, spring_f, spring_f * 0.985, wave="sine", curve="exp")
        spring *= _ac_delayed_decay(
            duration,
            0.007,
            max(0.008, duration / 7.0),
            attack_s=0.0003,
        )

        layers.append((spring, float(spring_gain)))

    # ------------------------------------------------------------------
    # Air / friction tail
    # ------------------------------------------------------------------
    air = lowpass(_ac_brown_noise(rng, duration), 900.0)
    air *= _ac_delayed_decay(
        duration,
        0.006,
        max(0.010, duration / 5.0),
        attack_s=0.0003,
    )
    air *= 0.07

    layers.append((air, 1.0))

    y = mix(layers)

    if highpass_cutoff > 0.0:
        y = highpass(y, highpass_cutoff)

    if lowpass_cutoff > 0.0:
        y = lowpass(y, min(lowpass_cutoff, SAMPLE_RATE / 2.0 - 300.0))

    y = _ac_softsat(y, max(1.1, drive * 0.35))

    return _ac_final(y)


def _ac_modal_instrument(
    rng: np.random.Generator,
    duration: float,
    base: float,
    ratios: List[float],
    gains: List[float],
    taus: List[float],
    exciter_low: float,
    exciter_high: float,
    exciter_gain: float = 0.55,
    body_bands: List[Tuple[float, float, float]] | None = None,
    body_gain: float = 0.45,
    drive: float = 2.2,
    lowpass_cutoff: float = 10000.0,
    highpass_cutoff: float = 50.0,
) -> np.ndarray:
    """
    Generic resonant acoustic object:
      - modal resonators
      - physical impact exciter
      - hollow-body cavity
      - friction tail
    """
    if body_bands is None:
        body_bands = [
            (300.0, 100.0, 0.50),
            (650.0, 180.0, 0.30),
        ]

    modal = _ac_modal(
        duration,
        base,
        ratios,
        gains,
        taus,
        rng=rng,
        detune=0.003,
    )

    impact_fm = _ac_fm_impact(
        rng,
        duration,
        exciter_high,
        exciter_low * 0.8,
        mod_ratio=2.2,
        index0=5.2,
        drive=drive * 0.80,
    )

    impact_noise = _ac_noise_burst(
        rng,
        duration,
        low=exciter_low,
        high=exciter_high,
        drive=drive * 0.70,
    )

    exciter = mix(
        [
            (impact_fm, 0.70),
            (impact_noise, exciter_gain),
        ]
    ) * _ac_delayed_decay(duration, 0.0, 0.004, attack_s=0.00015)

    cavity = _ac_cavity(rng, duration, body_bands, drive=drive * 0.55)
    cavity *= _ac_delayed_decay(
        duration,
        0.003,
        max(0.012, duration / 3.5),
        attack_s=0.0003,
    )
    cavity *= float(body_gain)

    friction = lowpass(_ac_brown_noise(rng, duration), 1200.0)
    friction *= _ac_delayed_decay(
        duration,
        0.005,
        max(0.010, duration / 5.0),
        attack_s=0.0003,
    )
    friction *= 0.08

    y = mix(
        [
            (modal, 0.88),
            (exciter, 1.0),
            (cavity, 1.0),
            (friction, 1.0),
        ]
    )

    if highpass_cutoff > 0.0:
        y = highpass(y, highpass_cutoff)

    if lowpass_cutoff > 0.0:
        y = lowpass(y, min(lowpass_cutoff, SAMPLE_RATE / 2.0 - 300.0))

    y = _ac_softsat(y, max(1.2, drive * 0.35))

    return _ac_final(y)


def _ac_ks_pluck(
    rng: np.random.Generator,
    freq: float,
    duration: float,
    brightness: float = 0.5,
    damping: float | None = None,
    dispersion: float = 0.20,
) -> np.ndarray:
    """
    Karplus-Strong plucked string with a simple first-order dispersion allpass.

    The allpass adds subtle phase dispersion, making nylon/string attacks less
    "delay-line synthetic" and more physically flexible.
    """
    freq = max(20.0, float(freq))
    brightness = float(np.clip(brightness, 0.0, 1.0))

    n_out = max(1, int(round(duration * SAMPLE_RATE)))
    period = max(2, int(round(SAMPLE_RATE / freq)))

    # Initial excitation: low-smoothed noise burst.
    buf = rng.uniform(-1.0, 1.0, period)

    for _ in range(2):
        buf = 0.5 * (buf + np.roll(buf, 1))

    if damping is None:
        damping = 0.986 + 0.012 * brightness

    damping = float(np.clip(damping, 0.970, 0.999))
    a = float(np.clip(dispersion, 0.0, 0.55))

    out = np.empty(n_out, dtype=np.float64)

    idx = 0
    prev_x = 0.0
    prev_y = 0.0

    for i in range(n_out):
        x = float(buf[idx])

        # First-order allpass:
        # y[n] = a*x[n] + x[n-1] - a*y[n-1]
        y_ap = a * x + prev_x - a * prev_y
        prev_x = x
        prev_y = y_ap

        nxt = (idx + 1) % period

        # Lowpass / averaging string junction.
        buf[idx] = damping * 0.5 * (y_ap + float(buf[nxt]))

        out[i] = x
        idx = nxt

    out -= float(np.mean(out))
    return out


def _ac_piano_note(
    rng: np.random.Generator,
    freq: float,
    duration: float,
    brightness: float = 1.0,
    hammer_gain: float = 0.42,
    body_gain: float = 0.22,
) -> np.ndarray:
    """
    Short piano note:
      - inharmonic steel string partials
      - felt hammer compression transient
      - soundboard / cavity resonance
    """
    beta = 0.00042

    ratios: List[float] = []
    gains: List[float] = []
    taus: List[float] = []

    base_gains = [
        1.00,
        0.52,
        0.30,
        0.18,
        0.11,
        0.065,
    ]

    for n in range(1, 7):
        ratio = n * (1.0 + beta * n * n)
        gain = base_gains[n - 1] * (0.72 + 0.28 * float(brightness))
        tau = duration / (2.1 + 0.55 * n)

        ratios.append(ratio)
        gains.append(gain)
        taus.append(tau)

    strings = _ac_modal(
        duration,
        freq,
        ratios,
        gains,
        taus,
        rng=rng,
        detune=0.002,
    )

    hammer_high = min(12000.0, max(900.0, freq * 7.0))
    hammer_low = max(120.0, freq * 1.3)

    hammer_fm = _ac_fm_impact(
        rng,
        duration,
        hammer_high,
        hammer_low,
        mod_ratio=1.7,
        index0=4.6,
        drive=2.3,
    )

    hammer_noise = _ac_noise_burst(
        rng,
        duration,
        low=hammer_low,
        high=min(12000.0, hammer_high * 1.2),
        drive=2.1,
    )

    hammer = mix(
        [
            (hammer_fm, 0.68),
            (hammer_noise, 0.55),
        ]
    ) * _ac_delayed_decay(duration, 0.0, 0.003, attack_s=0.00015)

    hammer *= float(hammer_gain)

    soundboard = _ac_cavity(
        rng,
        duration,
        [
            (110.0, 45.0, 0.50),
            (320.0, 95.0, 0.40),
            (560.0, 150.0, 0.30),
        ],
        drive=1.6,
    )

    soundboard *= _ac_delayed_decay(
        duration,
        0.003,
        max(0.012, duration / 3.2),
        attack_s=0.0003,
    )

    soundboard *= float(body_gain)

    y = mix(
        [
            (strings, 0.86),
            (hammer, 1.0),
            (soundboard, 1.0),
        ]
    )

    y = lowpass(y, 8200.0)
    y = highpass(y, 35.0)
    y = _ac_softsat(y, 1.45)

    return _ac_final(y)


def _ac_guitar_note(
    rng: np.random.Generator,
    freq: float,
    duration: float,
    brightness: float = 0.55,
    body_gain: float = 0.35,
    pick_gain: float = 0.28,
) -> np.ndarray:
    """
    Nylon guitar pluck:
      - Karplus-Strong string
      - dispersion allpass
      - guitar wood cavity resonances around 105 Hz and 220 Hz
      - soft finger/pick transient
    """
    string = _ac_ks_pluck(
        rng,
        freq,
        duration,
        brightness=brightness,
        dispersion=0.22,
    )

    string = lowpass(string, 5200.0)
    string = highpass(string, 60.0)

    body_air = _ac_cavity(
        rng,
        duration,
        [
            (105.0, 24.0, 0.75),
            (220.0, 48.0, 0.55),
        ],
        drive=1.6,
    )

    body_air *= _ac_delayed_decay(
        duration,
        0.003,
        max(0.012, duration / 3.6),
        attack_s=0.0003,
    )

    body_air *= 0.45

    string_body = mix(
        [
            (bandpass(string, 92.0, 128.0, order=2), 0.50),
            (bandpass(string, 185.0, 255.0, order=2), 0.40),
        ]
    )

    string_body *= float(body_gain)

    pick = _ac_noise_burst(
        rng,
        duration,
        low=1800.0,
        high=5200.0,
        drive=2.0,
    )

    pick *= _ac_delayed_decay(duration, 0.0, 0.0025, attack_s=0.00012)
    pick *= float(pick_gain)

    y = mix(
        [
            (string, 0.92),
            (body_air, 1.0),
            (string_body, 1.0),
            (pick, 1.0),
        ]
    )

    y = lowpass(y, 6400.0)
    y = highpass(y, 52.0)
    y = _ac_softsat(y, 1.4)

    return _ac_final(y)


def _ac_pizz_note(
    rng: np.random.Generator,
    freq: float,
    duration: float,
    brightness: float = 0.68,
    chamber_gain: float = 0.35,
) -> np.ndarray:
    """
    Orchestral pizzicato:
      - flesh finger pluck noise burst
      - coupled string vibration
      - wooden chamber resonance
    """
    string = _ac_ks_pluck(
        rng,
        freq,
        duration,
        brightness=brightness,
        dispersion=0.16,
    )

    string = lowpass(string, 7200.0)
    string = highpass(string, 75.0)

    finger = _ac_noise_burst(
        rng,
        duration,
        low=700.0,
        high=2600.0,
        drive=2.0,
    )

    finger *= _ac_delayed_decay(duration, 0.0, 0.003, attack_s=0.00015)
    finger *= 0.45

    body = _ac_cavity(
        rng,
        duration,
        [
            (260.0, 85.0, 0.55),
            (480.0, 150.0, 0.38),
        ],
        drive=1.6,
    )

    body *= _ac_delayed_decay(
        duration,
        0.003,
        max(0.012, duration / 3.5),
        attack_s=0.0003,
    )

    body *= float(chamber_gain)

    wood = lowpass(_ac_brown_noise(rng, duration), 850.0)
    wood *= _ac_delayed_decay(
        duration,
        0.004,
        max(0.010, duration / 4.5),
        attack_s=0.0003,
    )
    wood *= 0.12

    y = mix(
        [
            (string, 0.92),
            (finger, 1.0),
            (body, 1.0),
            (wood, 1.0),
        ]
    )

    y = lowpass(y, 7600.0)
    y = highpass(y, 65.0)
    y = _ac_softsat(y, 1.45)

    return _ac_final(y)


# ----------------------------------------------------------------------------
# 1. Deep Thock — Gateron Oil King
# ----------------------------------------------------------------------------

def build_gateron_oil_king(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Muffled, deep, lubricated downstroke with low-frequency acoustic cavity pop.
    """

    def thock(
        r: np.random.Generator,
        duration: float,
        bottom: float,
        cavity_a: float,
        cavity_b: float,
        click_gain: float = 0.22,
        spring_gain: float = 0.03,
        lp: float = 3200.0,
    ) -> np.ndarray:
        return _ac_switch_hit(
            r,
            duration,
            bottom=bottom,
            cavity_bands=[
                (cavity_a, 85.0, 0.72),
                (cavity_b, 140.0, 0.45),
            ],
            click_low=1100.0,
            click_high=2500.0,
            click_gain=click_gain,
            bottom_gain=0.96,
            spring_freq=820.0,
            spring_gain=spring_gain,
            drive=2.5,
            lowpass_cutoff=lp,
            highpass_cutoff=32.0,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        bottom = float(r.uniform(125.0, 180.0))
        duration = float(r.uniform(0.062, 0.082))
        cavity_a = float(r.uniform(180.0, 260.0))
        cavity_b = float(r.uniform(320.0, 430.0))
        click_gain = float(r.uniform(0.16, 0.26))
        spring_gain = float(r.uniform(0.02, 0.04))
        lp = float(r.uniform(2800.0, 3600.0))

        return thock(
            r,
            duration,
            bottom,
            cavity_a,
            cavity_b,
            click_gain,
            spring_gain,
            lp,
        )

    r_space = child_rng(rng, 20)
    space = thock(
        r_space,
        0.105,
        float(r_space.uniform(80.0, 100.0)),
        150.0,
        285.0,
        click_gain=0.15,
        spring_gain=0.02,
        lp=2500.0,
    )

    r_delete = child_rng(rng, 21)
    delete = thock(
        r_delete,
        0.042,
        225.0,
        330.0,
        520.0,
        click_gain=0.30,
        spring_gain=0.035,
        lp=4300.0,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = thock(
        r_ret_a,
        0.080,
        135.0,
        210.0,
        330.0,
        click_gain=0.20,
        spring_gain=0.03,
        lp=3100.0,
    )

    ret_b = thock(
        r_ret_b,
        0.046,
        210.0,
        320.0,
        500.0,
        click_gain=0.26,
        spring_gain=0.03,
        lp=3900.0,
    )

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.020, 0.75))

    shift = thock(
        child_rng(rng, 24),
        0.035,
        320.0,
        430.0,
        620.0,
        click_gain=0.32,
        spring_gain=0.04,
        lp=4800.0,
    )

    symbol_a = thock(
        child_rng(rng, 25),
        0.035,
        205.0,
        310.0,
        470.0,
        click_gain=0.26,
        spring_gain=0.03,
        lp=3700.0,
    )

    symbol_b = thock(
        child_rng(rng, 26),
        0.032,
        260.0,
        370.0,
        560.0,
        click_gain=0.28,
        spring_gain=0.03,
        lp=4100.0,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 2. Crisp Click — Kailh Box Jade
# ----------------------------------------------------------------------------

def build_kailh_box_jade(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Ultra-crisp dual-click tactile mechanism with high-Q resonance.
    """

    def crisp(
        r: np.random.Generator,
        duration: float,
        low: float,
        high: float,
        bottom: float = 320.0,
        second_delay: float = 0.0,
        second_gain: float = 0.0,
        lp: float = 9800.0,
        click_gain: float = 0.78,
    ) -> np.ndarray:
        return _ac_switch_hit(
            r,
            duration,
            bottom=bottom,
            cavity_bands=[
                (4200.0, 420.0, 0.32),
                (5300.0, 520.0, 0.28),
            ],
            click_low=low,
            click_high=high,
            click_gain=click_gain,
            bottom_gain=0.35,
            spring_freq=5200.0,
            spring_gain=0.05,
            drive=3.0,
            lowpass_cutoff=lp,
            highpass_cutoff=850.0,
            second_click_delay=second_delay,
            second_click_gain=second_gain,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.045, 0.058))
        low = float(r.uniform(3800.0, 4500.0))
        high = float(r.uniform(5000.0, 5500.0))
        bottom = float(r.uniform(280.0, 380.0))
        second_delay = float(r.uniform(0.010, 0.014))
        second_gain = float(r.uniform(0.70, 0.80))

        return crisp(
            r,
            duration,
            low,
            high,
            bottom=bottom,
            second_delay=second_delay,
            second_gain=second_gain,
        )

    r_space = child_rng(rng, 20)
    space = crisp(
        r_space,
        0.085,
        2500.0,
        4000.0,
        bottom=220.0,
        second_delay=0.018,
        second_gain=0.55,
        lp=8200.0,
        click_gain=0.65,
    )

    r_delete = child_rng(rng, 21)
    delete = crisp(
        r_delete,
        0.036,
        4500.0,
        6200.0,
        bottom=430.0,
        second_delay=0.0,
        second_gain=0.0,
        lp=10500.0,
        click_gain=0.82,
    )

    r_return = child_rng(rng, 22)
    ret = crisp(
        r_return,
        0.090,
        3600.0,
        5200.0,
        bottom=300.0,
        second_delay=0.024,
        second_gain=0.85,
        lp=9600.0,
        click_gain=0.75,
    )

    shift = crisp(
        child_rng(rng, 23),
        0.033,
        5000.0,
        6500.0,
        bottom=480.0,
        lp=10800.0,
        click_gain=0.85,
    )

    symbol_a = crisp(
        child_rng(rng, 24),
        0.032,
        4700.0,
        6100.0,
        bottom=430.0,
        lp=10500.0,
        click_gain=0.80,
    )

    symbol_b = crisp(
        child_rng(rng, 25),
        0.030,
        5200.0,
        6600.0,
        bottom=470.0,
        lp=11000.0,
        click_gain=0.80,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 3. Tactile Pop — Holy Panda
# ----------------------------------------------------------------------------

def build_holy_panda(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Rounded tactile bump followed by a sharp bottom-out pop.
    """

    def pop(
        r: np.random.Generator,
        duration: float,
        bottom: float,
        low: float,
        high: float,
        second_delay: float,
        second_gain: float,
        lp: float = 6500.0,
    ) -> np.ndarray:
        return _ac_switch_hit(
            r,
            duration,
            bottom=bottom,
            cavity_bands=[
                (240.0, 90.0, 0.55),
                (390.0, 130.0, 0.38),
            ],
            click_low=low,
            click_high=high,
            click_gain=0.48,
            bottom_gain=0.88,
            spring_freq=1250.0,
            spring_gain=0.06,
            drive=2.7,
            lowpass_cutoff=lp,
            highpass_cutoff=70.0,
            second_click_delay=second_delay,
            second_click_gain=second_gain,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.060, 0.080))
        bottom = float(r.uniform(225.0, 320.0))
        low = float(r.uniform(1400.0, 2000.0))
        high = float(r.uniform(2500.0, 3400.0))
        second_delay = float(r.uniform(0.011, 0.015))
        second_gain = float(r.uniform(0.58, 0.66))

        return pop(
            r,
            duration,
            bottom,
            low,
            high,
            second_delay,
            second_gain,
        )

    r_space = child_rng(rng, 20)
    space = pop(
        r_space,
        0.100,
        155.0,
        1000.0,
        2100.0,
        second_delay=0.016,
        second_gain=0.50,
        lp=5500.0,
    )

    r_delete = child_rng(rng, 21)
    delete = pop(
        r_delete,
        0.040,
        380.0,
        2100.0,
        3800.0,
        second_delay=0.0,
        second_gain=0.0,
        lp=7500.0,
    )

    r_return = child_rng(rng, 22)
    ret = pop(
        r_return,
        0.090,
        245.0,
        1500.0,
        2800.0,
        second_delay=0.020,
        second_gain=0.72,
        lp=6200.0,
    )

    shift = pop(
        child_rng(rng, 23),
        0.034,
        430.0,
        2300.0,
        4100.0,
        second_delay=0.0,
        second_gain=0.0,
        lp=7800.0,
    )

    symbol_a = pop(
        child_rng(rng, 24),
        0.034,
        330.0,
        1900.0,
        3300.0,
        second_delay=0.0,
        second_gain=0.0,
        lp=7000.0,
    )

    symbol_b = pop(
        child_rng(rng, 25),
        0.032,
        390.0,
        2200.0,
        3700.0,
        second_delay=0.0,
        second_gain=0.0,
        lp=7400.0,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 4. Mechanical Click — IBM Model M
# ----------------------------------------------------------------------------

def build_ibm_model_m(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Buckling spring snap with metallic ping and heavy solenoid/frame thud.
    """

    def model_m_hit(
        r: np.random.Generator,
        duration: float,
        thud: float,
        click_low: float,
        click_high: float,
        metal_gain: float = 0.35,
        second_delay: float = 0.008,
        second_gain: float = 0.45,
        lp: float = 10500.0,
    ) -> np.ndarray:
        base = _ac_switch_hit(
            r,
            duration,
            bottom=thud,
            cavity_bands=[
                (360.0, 130.0, 0.42),
                (950.0, 240.0, 0.30),
            ],
            click_low=click_low,
            click_high=click_high,
            click_gain=0.62,
            bottom_gain=0.95,
            spring_freq=2350.0,
            spring_gain=0.12,
            drive=3.2,
            lowpass_cutoff=lp,
            highpass_cutoff=65.0,
            second_click_delay=second_delay,
            second_click_gain=second_gain,
        )

        metal_freq = float(r.uniform(2600.0, 3300.0))

        metal = _ac_modal(
            duration,
            metal_freq,
            [1.0, 1.42, 1.92, 2.45],
            [0.30, 0.20, 0.14, 0.10],
            [
                duration / 5.5,
                duration / 6.5,
                duration / 7.5,
                duration / 8.5,
            ],
            rng=r,
            detune=0.005,
        )

        metal *= _ac_delayed_decay(
            duration,
            0.004,
            max(0.010, duration / 6.5),
            attack_s=0.0002,
        )

        metal *= float(metal_gain)

        y = mix(
            [
                (base, 1.0),
                (metal, 1.0),
            ]
        )

        return _ac_final(y)

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.080, 0.100))
        thud = float(r.uniform(75.0, 100.0))
        click_low = float(r.uniform(2400.0, 3000.0))
        click_high = float(r.uniform(4800.0, 5600.0))
        metal_gain = float(r.uniform(0.30, 0.40))

        return model_m_hit(
            r,
            duration,
            thud,
            click_low,
            click_high,
            metal_gain=metal_gain,
        )

    r_space = child_rng(rng, 20)
    space = model_m_hit(
        r_space,
        0.105,
        float(r_space.uniform(58.0, 72.0)),
        1800.0,
        4200.0,
        metal_gain=0.28,
        second_delay=0.010,
        second_gain=0.50,
        lp=9200.0,
    )

    r_delete = child_rng(rng, 21)
    delete = model_m_hit(
        r_delete,
        0.045,
        120.0,
        3200.0,
        6200.0,
        metal_gain=0.40,
        second_delay=0.0,
        second_gain=0.25,
        lp=11000.0,
    )

    r_return = child_rng(rng, 22)
    ret = model_m_hit(
        r_return,
        0.100,
        85.0,
        2200.0,
        5000.0,
        metal_gain=0.40,
        second_delay=0.018,
        second_gain=0.60,
        lp=10000.0,
    )

    shift = model_m_hit(
        child_rng(rng, 23),
        0.036,
        160.0,
        3500.0,
        6500.0,
        metal_gain=0.50,
        second_delay=0.0,
        second_gain=0.20,
        lp=11500.0,
    )

    symbol_a = model_m_hit(
        child_rng(rng, 24),
        0.034,
        150.0,
        3300.0,
        6200.0,
        metal_gain=0.45,
        second_delay=0.0,
        second_gain=0.20,
        lp=11200.0,
    )

    symbol_b = model_m_hit(
        child_rng(rng, 25),
        0.032,
        180.0,
        3700.0,
        6600.0,
        metal_gain=0.45,
        second_delay=0.0,
        second_gain=0.20,
        lp=11600.0,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.011))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 5. Typewriter — Classic Royal
# ----------------------------------------------------------------------------

def build_royal_typewriter(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Cast-iron typebar strike, ratchet spacebar, and mechanical bell return.
    """

    def typebar_hit(
        r: np.random.Generator,
        duration: float,
        base: float = 1.0,
        metal_gain: float = 0.45,
    ) -> np.ndarray:
        impact = _ac_fm_impact(
            r,
            duration,
            3800.0 * base,
            1200.0 * base,
            mod_ratio=2.1,
            index0=5.5,
            drive=3.0,
        )

        impact_noise = _ac_noise_burst(
            r,
            duration,
            low=1800.0 * base,
            high=6500.0 * base,
            drive=2.6,
        )

        strike = mix(
            [
                (impact, 0.70),
                (impact_noise, 0.60),
            ]
        ) * _ac_delayed_decay(duration, 0.0, 0.004, attack_s=0.00015)

        body = _ac_cavity(
            r,
            duration,
            [
                (320.0, 110.0, 0.50),
                (720.0, 200.0, 0.35),
            ],
            drive=1.8,
        )

        body *= _ac_delayed_decay(
            duration,
            0.003,
            max(0.012, duration / 4.0),
            attack_s=0.0003,
        )

        body *= 0.5

        metal = _ac_modal(
            duration,
            2800.0 * base,
            [1.0, 1.55, 2.18, 2.95],
            [0.35, 0.22, 0.15, 0.10],
            [
                duration / 5.0,
                duration / 6.0,
                duration / 7.0,
                duration / 8.0,
            ],
            rng=r,
            detune=0.005,
        )

        metal *= _ac_delayed_decay(
            duration,
            0.002,
            max(0.010, duration / 6.0),
            attack_s=0.0002,
        )

        metal *= float(metal_gain)

        y = mix(
            [
                (strike, 1.0),
                (body, 1.0),
                (metal, 1.0),
            ]
        )

        y = highpass(y, 220.0)
        y = lowpass(y, 10000.0)
        y = _ac_softsat(y, 1.6)

        return _ac_final(y)

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.050, 0.070))
        base = float(r.uniform(0.90, 1.10))
        metal_gain = float(r.uniform(0.38, 0.50))

        return typebar_hit(r, duration, base=base, metal_gain=metal_gain)

    def space_ratchet() -> np.ndarray:
        r = child_rng(rng, 20)
        duration = 0.105

        layers: List[Tuple[np.ndarray, float]] = []

        carriage = _ac_fm_impact(
            r,
            duration,
            190.0,
            65.0,
            mod_ratio=1.2,
            index0=3.5,
            drive=2.2,
        )

        carriage *= _ac_delayed_decay(duration, 0.0, 0.035, attack_s=0.0003)
        carriage *= 0.65

        layers.append((carriage, 1.0))

        for i, delay in enumerate([0.0, 0.024, 0.048]):
            rr = child_rng(r, 30 + i)

            tick_noise = _ac_noise_burst(
                rr,
                duration,
                low=1600.0 + 220.0 * i,
                high=4200.0 + 320.0 * i,
                drive=2.6,
            )

            tick_click = _ac_fm_impact(
                rr,
                duration,
                3600.0 + 250.0 * i,
                1800.0 + 150.0 * i,
                mod_ratio=2.0,
                index0=4.5,
                drive=2.8,
            )

            tick = mix(
                [
                    (tick_noise, 0.70),
                    (tick_click, 0.55),
                ]
            ) * _ac_delayed_decay(duration, delay, 0.006, attack_s=0.00015)

            layers.append((tick, 1.0))

        y = mix(layers)
        y = highpass(y, 120.0)
        y = lowpass(y, 8500.0)
        y = _ac_softsat(y, 1.5)

        return _ac_final(y)

    def bell_return() -> np.ndarray:
        r = child_rng(rng, 21)
        duration = 0.105

        bell = _ac_modal(
            duration,
            1760.0,
            [1.0, 2.0, 2.7, 3.6],
            [0.62, 0.36, 0.22, 0.13],
            [
                duration / 3.0,
                duration / 4.2,
                duration / 5.2,
                duration / 6.2,
            ],
            rng=r,
            detune=0.003,
        )

        strike_noise = _ac_noise_burst(
            r,
            duration,
            low=2600.0,
            high=7200.0,
            drive=2.8,
        ) * _ac_delayed_decay(duration, 0.0, 0.004, attack_s=0.00012)

        strike_fm = _ac_fm_impact(
            r,
            duration,
            4200.0,
            2100.0,
            mod_ratio=2.2,
            index0=5.0,
            drive=2.8,
        ) * _ac_delayed_decay(duration, 0.0, 0.003, attack_s=0.00012)

        body = _ac_cavity(
            r,
            duration,
            [
                (520.0, 160.0, 0.35),
                (1050.0, 260.0, 0.25),
            ],
            drive=1.5,
        )

        body *= _ac_delayed_decay(
            duration,
            0.003,
            max(0.012, duration / 4.0),
            attack_s=0.0003,
        )

        body *= 0.35

        y = mix(
            [
                (bell, 0.85),
                (strike_noise, 0.55),
                (strike_fm, 0.50),
                (body, 1.0),
            ]
        )

        y = highpass(y, 180.0)
        y = lowpass(y, 11500.0)
        y = _ac_softsat(y, 1.55)

        return _ac_final(y)

    r_delete = child_rng(rng, 22)
    delete = typebar_hit(
        r_delete,
        0.038,
        base=1.20,
        metal_gain=0.50,
    )

    shift = typebar_hit(
        child_rng(rng, 23),
        0.034,
        base=1.35,
        metal_gain=0.52,
    )

    symbol_a = typebar_hit(
        child_rng(rng, 24),
        0.034,
        base=1.10,
        metal_gain=0.46,
    )

    symbol_b = typebar_hit(
        child_rng(rng, 25),
        0.032,
        base=1.28,
        metal_gain=0.48,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.011))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space_ratchet()], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([bell_return()], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 6. Creamy Linear — Jelly
# ----------------------------------------------------------------------------

def build_creamy_linear_jelly(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Soft, ultra-smooth dampened linear switch bottom-out.
    """

    def jelly(
        r: np.random.Generator,
        duration: float,
        bottom: float,
        cavity_a: float,
        cavity_b: float,
        click_gain: float = 0.12,
        drive: float = 1.8,
    ) -> np.ndarray:
        return _ac_switch_hit(
            r,
            duration,
            bottom=bottom,
            cavity_bands=[
                (cavity_a, 70.0, 0.70),
                (cavity_b, 110.0, 0.45),
            ],
            click_low=160.0,
            click_high=520.0,
            click_gain=click_gain,
            bottom_gain=0.95,
            spring_freq=320.0,
            spring_gain=0.0,
            drive=drive,
            lowpass_cutoff=650.0,
            highpass_cutoff=24.0,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.065, 0.090))
        bottom = float(r.uniform(75.0, 120.0))
        cavity_a = float(r.uniform(160.0, 220.0))
        cavity_b = float(r.uniform(280.0, 360.0))
        click_gain = float(r.uniform(0.08, 0.14))
        drive = float(r.uniform(1.6, 1.9))

        return jelly(
            r,
            duration,
            bottom,
            cavity_a,
            cavity_b,
            click_gain=click_gain,
            drive=drive,
        )

    r_space = child_rng(rng, 20)
    space = jelly(
        r_space,
        0.105,
        float(r_space.uniform(60.0, 75.0)),
        140.0,
        260.0,
        click_gain=0.10,
        drive=1.7,
    )

    r_delete = child_rng(rng, 21)
    delete = jelly(
        r_delete,
        0.045,
        150.0,
        230.0,
        380.0,
        click_gain=0.16,
        drive=1.8,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = jelly(
        r_ret_a,
        0.080,
        85.0,
        170.0,
        290.0,
        click_gain=0.11,
        drive=1.7,
    )

    ret_b = jelly(
        r_ret_b,
        0.045,
        120.0,
        210.0,
        340.0,
        click_gain=0.13,
        drive=1.8,
    )

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.020, 0.72))

    shift = jelly(
        child_rng(rng, 24),
        0.036,
        190.0,
        260.0,
        420.0,
        click_gain=0.16,
        drive=1.9,
    )

    symbol_a = jelly(
        child_rng(rng, 25),
        0.035,
        135.0,
        210.0,
        330.0,
        click_gain=0.13,
        drive=1.8,
    )

    symbol_b = jelly(
        child_rng(rng, 26),
        0.033,
        165.0,
        240.0,
        370.0,
        click_gain=0.14,
        drive=1.8,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 7. Woodblock — Teak
# ----------------------------------------------------------------------------

def build_teak_woodblock(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Acoustic wooden mallet resonance with dual wood-grain modes.
    """

    def wood_hit(
        r: np.random.Generator,
        duration: float,
        base: float,
        brightness: float = 1.0,
        body_low: float = 320.0,
        body_high: float = 720.0,
    ) -> np.ndarray:
        ratios = [1.0, 2.07, 2.92]
        gains = [0.85, 0.45, 0.22]
        taus = [
            duration / 4.2,
            duration / 5.5,
            duration / 6.8,
        ]

        return _ac_modal_instrument(
            r,
            duration,
            base,
            ratios,
            gains,
            taus,
            exciter_low=1800.0 * brightness,
            exciter_high=5600.0 * brightness,
            exciter_gain=0.55,
            body_bands=[
                (body_low, 110.0, 0.38),
                (body_high, 190.0, 0.28),
            ],
            body_gain=0.42,
            drive=2.4,
            lowpass_cutoff=9500.0,
            highpass_cutoff=120.0,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.045, 0.062))
        base = float(r.uniform(680.0, 780.0))
        brightness = float(r.uniform(0.90, 1.10))

        return wood_hit(r, duration, base, brightness=brightness)

    r_space = child_rng(rng, 20)
    space = wood_hit(
        r_space,
        0.100,
        390.0,
        brightness=0.85,
        body_low=220.0,
        body_high=420.0,
    )

    r_delete = child_rng(rng, 21)
    delete = wood_hit(
        r_delete,
        0.036,
        1300.0,
        brightness=1.25,
        body_low=420.0,
        body_high=900.0,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = wood_hit(r_ret_a, 0.055, 720.0, brightness=1.0)
    ret_b = wood_hit(r_ret_b, 0.050, 540.0, brightness=0.92)

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.018, 0.85))

    shift = wood_hit(
        child_rng(rng, 24),
        0.034,
        1700.0,
        brightness=1.35,
        body_low=500.0,
        body_high=1050.0,
    )

    symbol_a = wood_hit(
        child_rng(rng, 25),
        0.035,
        900.0,
        brightness=1.05,
        body_low=360.0,
        body_high=780.0,
    )

    symbol_b = wood_hit(
        child_rng(rng, 26),
        0.032,
        1180.0,
        brightness=1.18,
        body_low=420.0,
        body_high=920.0,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 8. Piano — Acoustic Grand
# ----------------------------------------------------------------------------

def build_grand_piano(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Short acoustic grand piano key: felt hammer, inharmonic strings, soundboard.
    """

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        freq = float(
            r.choice(
                [
                    261.63,
                    293.66,
                    329.63,
                    349.23,
                    392.00,
                ]
            )
        ) * float(r.uniform(0.995, 1.005))

        duration = float(r.uniform(0.075, 0.095))
        brightness = float(r.uniform(0.85, 1.05))

        return _ac_piano_note(
            r,
            freq,
            duration,
            brightness=brightness,
            hammer_gain=float(r.uniform(0.36, 0.46)),
            body_gain=float(r.uniform(0.18, 0.26)),
        )

    space = _ac_piano_note(
        child_rng(rng, 20),
        98.0,
        0.105,
        brightness=0.80,
        hammer_gain=0.30,
        body_gain=0.28,
    )

    r_delete = child_rng(rng, 21)
    delete = _ac_piano_note(
        r_delete,
        float(r_delete.uniform(820.0, 940.0)),
        0.040,
        brightness=1.15,
        hammer_gain=0.50,
        body_gain=0.10,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = _ac_piano_note(
        r_ret_a,
        392.0,
        0.085,
        brightness=0.95,
        hammer_gain=0.40,
        body_gain=0.18,
    )

    ret_b = _ac_piano_note(
        r_ret_b,
        587.33,
        0.065,
        brightness=1.00,
        hammer_gain=0.38,
        body_gain=0.15,
    )

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.018, 0.88))

    shift = _ac_piano_note(
        child_rng(rng, 24),
        1567.98,
        0.035,
        brightness=1.20,
        hammer_gain=0.52,
        body_gain=0.08,
    )

    symbol_a = _ac_piano_note(
        child_rng(rng, 25),
        987.77,
        0.035,
        brightness=1.05,
        hammer_gain=0.44,
        body_gain=0.10,
    )

    symbol_b = _ac_piano_note(
        child_rng(rng, 26),
        1174.66,
        0.035,
        brightness=1.05,
        hammer_gain=0.44,
        body_gain=0.10,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 9. Acoustic Pluck — Nylon Guitar
# ----------------------------------------------------------------------------

def build_nylon_guitar(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Karplus-Strong nylon string with wooden guitar cavity resonance.
    """

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        freq = float(
            r.choice(
                [
                    196.00,
                    246.94,
                    329.63,
                ]
            )
        ) * float(r.uniform(0.996, 1.004))

        duration = float(r.uniform(0.080, 0.100))
        brightness = float(r.uniform(0.45, 0.62))

        return _ac_guitar_note(
            r,
            freq,
            duration,
            brightness=brightness,
            body_gain=float(r.uniform(0.30, 0.40)),
            pick_gain=float(r.uniform(0.22, 0.30)),
        )

    space = _ac_guitar_note(
        child_rng(rng, 20),
        82.41,
        0.105,
        brightness=0.35,
        body_gain=0.50,
        pick_gain=0.16,
    )

    r_delete = child_rng(rng, 21)
    delete = _ac_guitar_note(
        r_delete,
        float(r_delete.uniform(659.0, 784.0)),
        0.042,
        brightness=0.80,
        body_gain=0.14,
        pick_gain=0.35,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = _ac_guitar_note(
        r_ret_a,
        196.0,
        0.070,
        brightness=0.52,
        body_gain=0.34,
        pick_gain=0.24,
    )

    ret_b = _ac_guitar_note(
        r_ret_b,
        293.66,
        0.055,
        brightness=0.58,
        body_gain=0.30,
        pick_gain=0.24,
    )

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.020, 0.88))

    shift = _ac_guitar_note(
        child_rng(rng, 24),
        1174.66,
        0.035,
        brightness=0.85,
        body_gain=0.12,
        pick_gain=0.35,
    )

    symbol_a = _ac_guitar_note(
        child_rng(rng, 25),
        880.0,
        0.035,
        brightness=0.72,
        body_gain=0.16,
        pick_gain=0.30,
    )

    symbol_b = _ac_guitar_note(
        child_rng(rng, 26),
        987.77,
        0.035,
        brightness=0.72,
        body_gain=0.16,
        pick_gain=0.30,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.008))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 10. Folk Drum — Kerala Chenda
# ----------------------------------------------------------------------------

def build_kerala_chenda(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Cane stick transient, parchment membrane modes, and wooden shell thud.
    """

    def chenda_hit(
        r: np.random.Generator,
        duration: float,
        base: float,
        stick_gain: float = 0.70,
        body_gain: float = 0.50,
    ) -> np.ndarray:
        ratios = [1.0, 1.59, 2.14, 2.30]
        gains = [0.80, 0.38, 0.26, 0.20]
        taus = [
            duration / 3.6,
            duration / 4.6,
            duration / 5.6,
            duration / 6.2,
        ]

        return _ac_modal_instrument(
            r,
            duration,
            base,
            ratios,
            gains,
            taus,
            exciter_low=2600.0,
            exciter_high=6800.0,
            exciter_gain=stick_gain,
            body_bands=[
                (max(45.0, base * 0.55), max(30.0, base * 0.35), 0.65),
                (max(90.0, base * 1.30), max(50.0, base * 0.50), 0.35),
            ],
            body_gain=body_gain,
            drive=2.6,
            lowpass_cutoff=9500.0,
            highpass_cutoff=70.0,
        )

    def rim_crack(
        r: np.random.Generator,
        duration: float,
        freq: float = 4200.0,
    ) -> np.ndarray:
        crack = _ac_noise_burst(
            r,
            duration,
            low=2500.0,
            high=7500.0,
            drive=2.8,
        ) * _ac_delayed_decay(duration, 0.0, duration / 2.8, attack_s=0.00012)

        click = _ac_fm_impact(
            r,
            duration,
            freq * 1.2,
            freq * 0.7,
            mod_ratio=2.0,
            index0=5.0,
            drive=2.5,
        ) * _ac_delayed_decay(duration, 0.0, 0.003, attack_s=0.00012)

        body = _ac_cavity(
            r,
            duration,
            [
                (400.0, 150.0, 0.40),
            ],
            drive=1.5,
        )

        body *= _ac_delayed_decay(duration, 0.002, duration / 5.0, attack_s=0.0002)
        body *= 0.30

        y = mix(
            [
                (crack, 0.80),
                (click, 0.65),
                (body, 1.0),
            ]
        )

        y = highpass(y, 200.0)
        y = lowpass(y, 10500.0)
        y = _ac_softsat(y, 1.55)

        return _ac_final(y)

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.055, 0.078))
        base = float(r.uniform(210.0, 280.0))
        stick_gain = float(r.uniform(0.65, 0.80))
        body_gain = float(r.uniform(0.42, 0.56))

        return chenda_hit(
            r,
            duration,
            base,
            stick_gain=stick_gain,
            body_gain=body_gain,
        )

    r_space = child_rng(rng, 20)
    space = chenda_hit(
        r_space,
        0.105,
        float(r_space.uniform(95.0, 112.0)),
        stick_gain=0.35,
        body_gain=0.65,
    )

    r_delete = child_rng(rng, 21)
    delete = rim_crack(
        r_delete,
        0.036,
        freq=float(r_delete.uniform(3900.0, 4700.0)),
    )

    r_ret_body = child_rng(rng, 22)
    r_ret_rim = child_rng(rng, 23)

    ret_body = chenda_hit(
        r_ret_body,
        0.085,
        150.0,
        stick_gain=0.48,
        body_gain=0.60,
    )

    ret_rim = rim_crack(
        r_ret_rim,
        0.032,
        freq=4300.0,
    )

    ret = _ac_final(add_delayed(ret_body, ret_rim, 0.022, 0.85))

    shift = rim_crack(
        child_rng(rng, 24),
        0.032,
        freq=4800.0,
    )

    symbol_a = chenda_hit(
        child_rng(rng, 25),
        0.038,
        320.0,
        stick_gain=0.68,
        body_gain=0.32,
    )

    symbol_b = rim_crack(
        child_rng(rng, 26),
        0.030,
        freq=4600.0,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.011))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 11. Resonant Drum — Carnatic Mridangam
# ----------------------------------------------------------------------------

def build_carnatic_mridangam(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Metallic harmonic ring and downward pitch-bending bass resonance.
    """

    def mrid_hit(
        r: np.random.Generator,
        duration: float,
        base: float,
        ring_gain: float = 0.55,
        bass_gain: float = 0.65,
        bend: bool = False,
        slap_gain: float = 0.45,
    ) -> np.ndarray:
        ring = _ac_modal(
            duration,
            base * 2.0,
            [1.0, 2.0, 3.0, 4.05],
            [0.62, 0.38, 0.22, 0.12],
            [
                duration / 3.8,
                duration / 4.6,
                duration / 5.6,
                duration / 6.8,
            ],
            rng=r,
            detune=0.003,
        )

        if bend:
            bass = tone(duration, base * 1.32, base * 0.70, wave="sine", curve="exp")
        else:
            bass = tone(duration, base * 1.06, base * 0.88, wave="sine", curve="exp")

        bass *= decay_env(duration, max(0.006, duration / 3.2), attack_s=0.0002)
        bass = _ac_softsat(bass, 1.9)

        slap = _ac_noise_burst(
            r,
            duration,
            low=1200.0,
            high=4600.0,
            drive=2.3,
        )

        slap *= _ac_delayed_decay(duration, 0.0, 0.004, attack_s=0.00015)
        slap *= float(slap_gain)

        body = _ac_cavity(
            r,
            duration,
            [
                (max(60.0, base * 1.1), max(35.0, base * 0.5), 0.55),
                (max(120.0, base * 2.3), max(60.0, base * 0.8), 0.35),
            ],
            drive=1.7,
        )

        body *= _ac_delayed_decay(
            duration,
            0.003,
            max(0.012, duration / 3.6),
            attack_s=0.0003,
        )

        body *= 0.45

        y = mix(
            [
                (ring, ring_gain),
                (bass, bass_gain),
                (slap, 1.0),
                (body, 1.0),
            ]
        )

        y = highpass(y, 45.0)
        y = lowpass(y, 8600.0)
        y = _ac_softsat(y, 1.55)

        return _ac_final(y)

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        duration = float(r.uniform(0.075, 0.095))
        base = float(r.uniform(180.0, 240.0))
        bend = bool(r.random() < 0.35)

        return mrid_hit(
            r,
            duration,
            base,
            ring_gain=float(r.uniform(0.48, 0.60)),
            bass_gain=float(r.uniform(0.58, 0.72)),
            bend=bend,
            slap_gain=float(r.uniform(0.38, 0.50)),
        )

    r_space = child_rng(rng, 20)
    space = mrid_hit(
        r_space,
        0.105,
        float(r_space.uniform(75.0, 95.0)),
        ring_gain=0.25,
        bass_gain=0.90,
        bend=True,
        slap_gain=0.22,
    )

    r_delete = child_rng(rng, 21)
    delete = mrid_hit(
        r_delete,
        0.040,
        float(r_delete.uniform(650.0, 780.0)),
        ring_gain=0.75,
        bass_gain=0.12,
        bend=False,
        slap_gain=0.55,
    )

    r_ret_low = child_rng(rng, 22)
    r_ret_high = child_rng(rng, 23)

    ret_low = mrid_hit(
        r_ret_low,
        0.095,
        100.0,
        ring_gain=0.22,
        bass_gain=0.88,
        bend=True,
        slap_gain=0.20,
    )

    ret_high = mrid_hit(
        r_ret_high,
        0.045,
        520.0,
        ring_gain=0.70,
        bass_gain=0.18,
        bend=False,
        slap_gain=0.48,
    )

    ret = _ac_final(add_delayed(ret_low, ret_high, 0.035, 0.85))

    shift = mrid_hit(
        child_rng(rng, 24),
        0.035,
        1200.0,
        ring_gain=0.82,
        bass_gain=0.10,
        bend=False,
        slap_gain=0.52,
    )

    symbol_a = mrid_hit(
        child_rng(rng, 25),
        0.035,
        720.0,
        ring_gain=0.72,
        bass_gain=0.12,
        bend=False,
        slap_gain=0.52,
    )

    symbol_b = mrid_hit(
        child_rng(rng, 26),
        0.040,
        140.0,
        ring_gain=0.25,
        bass_gain=0.78,
        bend=True,
        slap_gain=0.25,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.010))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 12. Kalimba — Thumb Piano
# ----------------------------------------------------------------------------

def build_kalimba_tines(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Plucked steel tines with cantilever modes and wooden soundbox resonance.
    """

    def kalimba_note(
        r: np.random.Generator,
        freq: float,
        duration: float,
        body_gain: float = 0.35,
        click_gain: float = 0.38,
    ) -> np.ndarray:
        ratios = [1.0, 2.76, 5.40]
        gains = [1.0, 0.28, 0.12]
        taus = [
            duration / 3.0,
            duration / 4.4,
            duration / 5.8,
        ]

        return _ac_modal_instrument(
            r,
            duration,
            freq,
            ratios,
            gains,
            taus,
            exciter_low=2600.0,
            exciter_high=7200.0,
            exciter_gain=click_gain,
            body_bands=[
                (280.0, 90.0, 0.50),
                (540.0, 160.0, 0.35),
            ],
            body_gain=body_gain,
            drive=2.0,
            lowpass_cutoff=11500.0,
            highpass_cutoff=120.0,
        )

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        freq = float(
            r.choice(
                [
                    523.25,
                    587.33,
                    659.25,
                    783.99,
                ]
            )
        ) * float(r.uniform(0.997, 1.003))

        duration = float(r.uniform(0.075, 0.095))

        return kalimba_note(
            r,
            freq,
            duration,
            body_gain=float(r.uniform(0.28, 0.38)),
            click_gain=float(r.uniform(0.32, 0.42)),
        )

    space = kalimba_note(
        child_rng(rng, 20),
        220.0,
        0.105,
        body_gain=0.50,
        click_gain=0.26,
    )

    delete = kalimba_note(
        child_rng(rng, 21),
        1567.98,
        0.036,
        body_gain=0.14,
        click_gain=0.48,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = kalimba_note(
        r_ret_a,
        523.25,
        0.085,
        body_gain=0.34,
        click_gain=0.34,
    )

    ret_b = kalimba_note(
        r_ret_b,
        783.99,
        0.060,
        body_gain=0.30,
        click_gain=0.32,
    )

    ret = _ac_final(add_delayed(ret_a, ret_b, 0.018, 0.88))

    shift = kalimba_note(
        child_rng(rng, 24),
        2093.0,
        0.032,
        body_gain=0.12,
        click_gain=0.48,
    )

    symbol_a = kalimba_note(
        child_rng(rng, 25),
        1046.5,
        0.034,
        body_gain=0.20,
        click_gain=0.40,
    )

    symbol_b = kalimba_note(
        child_rng(rng, 26),
        1174.66,
        0.034,
        body_gain=0.20,
        click_gain=0.40,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.009))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------
# 13. Pizzicato — Orchestral Strings
# ----------------------------------------------------------------------------

def build_orchestral_pizzicato(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Finger-plucked string with fast attack and acoustic chamber decay.
    """

    def default_variant(i: int) -> np.ndarray:
        r = child_rng(rng, 10 + i)

        freq = float(
            r.choice(
                [
                    293.66,
                    392.00,
                    587.33,
                ]
            )
        ) * float(r.uniform(0.996, 1.004))

        duration = float(r.uniform(0.070, 0.090))
        brightness = float(r.uniform(0.60, 0.80))
        chamber_gain = float(r.uniform(0.28, 0.38))

        return _ac_pizz_note(
            r,
            freq,
            duration,
            brightness=brightness,
            chamber_gain=chamber_gain,
        )

    space = _ac_pizz_note(
        child_rng(rng, 20),
        98.0,
        0.105,
        brightness=0.45,
        chamber_gain=0.50,
    )

    r_delete = child_rng(rng, 21)
    delete = _ac_pizz_note(
        r_delete,
        float(r_delete.uniform(1318.0, 1568.0)),
        0.036,
        brightness=0.88,
        chamber_gain=0.14,
    )

    r_ret_a = child_rng(rng, 22)
    r_ret_b = child_rng(rng, 23)

    ret_a = _ac_pizz_note(
        r_ret_a,
        392.0,
        0.085,
        brightness=0.70,
        chamber_gain=0.34,
    )

    ret_b = _ac_pizz_note(
        r_ret_b,
        587.33,
        0.085,
        brightness=0.72,
        chamber_gain=0.30,
    )

    ret = mix(
        [
            (ret_a, 0.80),
            (ret_b, 0.72),
        ]
    )

    ret = _ac_final(ret)

    shift = _ac_pizz_note(
        child_rng(rng, 24),
        2093.0,
        0.032,
        brightness=0.92,
        chamber_gain=0.12,
    )

    symbol_a = _ac_pizz_note(
        child_rng(rng, 25),
        987.77,
        0.034,
        brightness=0.80,
        chamber_gain=0.18,
    )

    symbol_b = _ac_pizz_note(
        child_rng(rng, 26),
        1318.51,
        0.034,
        brightness=0.80,
        chamber_gain=0.18,
    )

    symbol = _ac_final(concat([symbol_a, symbol_b], gap_s=0.009))

    return {
        "keypress.default": ev([default_variant(i) for i in range(3)], "random", 1.0),
        "keypress.space": ev([space], "single", 1.0),
        "keypress.delete": ev([delete], "single", 0.95),
        "keypress.return": ev([ret], "single", 0.95),
        "keypress.shift": ev([shift], "single", 0.90),
        "keypress.symbol": ev([symbol], "single", 0.90),
    }


# ----------------------------------------------------------------------------


# ==========================================
# 0. Soft Pudding Synth (Restored Procedural)
# ==========================================
def build_soft_pudding_synth(rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Algorithmically recreates the 'soft pudding' sound using DSP primitives.
    Characterized by a short, low-frequency, heavily dampened tap.
    """
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.6, wave="sine") * decay_env(duration, tau=0.01)
        click = lowpass(noise(rng, 0.003), 1000.0) * decay_env(0.003, tau=0.001)
        return finalize(mix([(body, 0.75), (click, 0.25)]), target_db=-3.0)

    return {
        "keypress.default": ev([hit(220.0, 0.04), hit(235.0, 0.04)], "random"),
        "keypress.space": ev([hit(120.0, 0.06)], "single"),
        "keypress.delete": ev([hit(320.0, 0.03)], "single"),
        "keypress.return": ev([hit(180.0, 0.05)], "single"),
    }


# ==========================================
# 1. Muted Marshmallow (Ultra-Dead Sub Thud)
# ==========================================
def build_muted_marshmallow(rng: np.random.Generator) -> Dict[str, Dict]:
    """Extremely short, heavy, and dead. Almost zero click, just a soft sub-bass thud."""
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.5, wave="sine") * decay_env(duration, tau=0.006)
        click = lowpass(noise(rng, 0.002), 600.0) * decay_env(0.002, tau=0.001)
        return finalize(mix([(body, 0.95), (click, 0.05)]), target_db=-4.0)

    return {
        "keypress.default": ev([hit(130.0, 0.03), hit(140.0, 0.03)], "random"),
        "keypress.space": ev([hit(80.0, 0.04)], "single"),
        "keypress.delete": ev([hit(180.0, 0.025)], "single"),
        "keypress.return": ev([hit(110.0, 0.035)], "single"),
    }


# ==========================================
# 2. Felted Thock (Enthusiast Dampened)
# ==========================================
def build_felted_thock(rng: np.random.Generator) -> Dict[str, Dict]:
    """Like a high-end mechanical keyboard heavily dampened with thick silicone/felt."""
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.8, wave="triangle") * decay_env(duration, tau=0.012)
        click = lowpass(noise(rng, 0.003), 1200.0) * decay_env(0.003, tau=0.002)
        return finalize(mix([(body, 0.85), (click, 0.15)]), target_db=-3.0)

    return {
        "keypress.default": ev([hit(260.0, 0.04), hit(275.0, 0.04)], "random"),
        "keypress.space": ev([hit(160.0, 0.05)], "single"),
        "keypress.delete": ev([hit(320.0, 0.03)], "single"),
        "keypress.return": ev([hit(200.0, 0.045)], "single"),
    }


# ==========================================
# 3. Membrane Squish (Retro Office Soft)
# ==========================================
def build_membrane_squish(rng: np.random.Generator) -> Dict[str, Dict]:
    """Vintage 90s rubber dome keyboard. Soft bottom-out with a tiny high-end tick."""
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.9, wave="sine") * decay_env(duration, tau=0.015)
        squish = bandpass(noise(rng, 0.004), 800.0, 1500.0) * decay_env(0.004, tau=0.003)
        tick = highpass(noise(rng, 0.001), 4000.0) * decay_env(0.001, tau=0.001)
        return finalize(mix([(body, 0.7), (squish, 0.25), (tick, 0.05)]), target_db=-2.5)

    return {
        "keypress.default": ev([hit(350.0, 0.04), hit(370.0, 0.04)], "random"),
        "keypress.space": ev([hit(220.0, 0.05)], "single"),
        "keypress.delete": ev([hit(450.0, 0.03)], "single"),
        "keypress.return": ev([hit(280.0, 0.045)], "single"),
    }


# ==========================================
# 4. Cork Tap (Organic Muted Wood)
# ==========================================
def build_cork_tap(rng: np.random.Generator) -> Dict[str, Dict]:
    """Sounds like tapping on a thick piece of cork or soft wood. Earthy and dry."""
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.7, wave="triangle") * decay_env(duration, tau=0.01)
        texture = bandpass(noise(rng, 0.005), 400.0, 2500.0) * decay_env(0.005, tau=0.002)
        return finalize(mix([(body, 0.6), (texture, 0.4)]), target_db=-3.5)

    return {
        "keypress.default": ev([hit(210.0, 0.035), hit(225.0, 0.035)], "random"),
        "keypress.space": ev([hit(140.0, 0.045)], "single"),
        "keypress.delete": ev([hit(300.0, 0.025)], "single"),
        "keypress.return": ev([hit(180.0, 0.04)], "single"),
    }


# ==========================================
# 5. Velvet Whisper (ASMR Airy Soft)
# ==========================================
def build_velvet_whisper(rng: np.random.Generator) -> Dict[str, Dict]:
    """Very smooth, high-end but low-volume. A soft, airy release with a quiet body."""
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.95, wave="sine") * decay_env(duration, tau=0.02)
        air = highpass(noise(rng, 0.008), 2500.0) * decay_env(0.008, tau=0.005)
        return finalize(mix([(body, 0.5), (air, 0.5)]), target_db=-5.0)

    return {
        "keypress.default": ev([hit(400.0, 0.04), hit(420.0, 0.04)], "random"),
        "keypress.space": ev([hit(280.0, 0.05)], "single"),
        "keypress.delete": ev([hit(500.0, 0.03)], "single"),
        "keypress.return": ev([hit(340.0, 0.045)], "single"),
    }

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
        slug="thock",
        name="Deep Thock",
        summary="Deep lubricated switch clack.",
        tags=["mechanical", "thock", "deep"],
        builder=build_gateron_oil_king,
        master_volume=0.85,
    ),
    PackSpec(
        slug="clicky",
        name="Crisp Click",
        summary="High-pitched sharp click.",
        tags=["clicky", "tactile", "bright"],
        builder=build_kailh_box_jade,
        master_volume=0.80,
    ),
    PackSpec(
        slug="tactile",
        name="Tactile Pop",
        summary="Snappy tactile bump and pop.",
        tags=["tactile", "snappy"],
        builder=build_holy_panda,
        master_volume=0.85,
    ),
    PackSpec(
        slug="mechanical",
        name="Mechanical Click",
        summary="Retro mechanical spring click.",
        tags=["mechanical", "retro", "click"],
        builder=build_ibm_model_m,
        master_volume=0.82,
    ),
    PackSpec(
        slug="typewriter",
        name="Typewriter",
        summary="Vintage carriage and chime.",
        tags=["typewriter", "vintage", "chime"],
        builder=build_royal_typewriter,
        master_volume=0.82,
    ),
    PackSpec(
        slug="creamy",
        name="Creamy Linear",
        summary="Soft dampened linear tap.",
        tags=["linear", "muted", "soft"],
        builder=build_creamy_linear_jelly,
        master_volume=0.88,
    ),
    PackSpec(
        slug="chiptune",
        name="8-Bit Chiptune",
        summary="Retro square-wave arcade blips.",
        tags=["retro", "chiptune", "8-bit"],
        builder=build_chiptune_arcade,
        master_volume=0.72,
    ),
    PackSpec(
        slug="glass",
        name="Glass Marble",
        summary="Polished mineral tap.",
        tags=["minimal", "glass", "clean"],
        builder=build_ceramic_glass_marble,
        master_volume=0.80,
    ),
    PackSpec(
        slug="bubble",
        name="Bubble Pop",
        summary="Soft liquid droplet burst.",
        tags=["liquid", "bubble", "soft"],
        builder=build_water_bubble_pop,
        master_volume=0.82,
    ),
    PackSpec(
        slug="woodblock",
        name="Woodblock",
        summary="Acoustic wooden mallet tap.",
        tags=["wood", "percussion", "acoustic"],
        builder=build_teak_woodblock,
        master_volume=0.85,
    ),
    PackSpec(
        slug="piano",
        name="Piano",
        summary="Warm harmonic key strike.",
        tags=["piano", "acoustic", "keys"],
        builder=build_grand_piano,
        master_volume=0.85,
    ),
    PackSpec(
        slug="acoustic-pluck",
        name="Acoustic Pluck",
        summary="Plucked nylon string tone.",
        tags=["guitar", "strings", "pluck"],
        builder=build_nylon_guitar,
        master_volume=0.85,
    ),
    PackSpec(
        slug="folk-drum",
        name="Folk Drum",
        summary="High-tension rim and drum hit.",
        tags=["percussion", "ethnic", "drum"],
        builder=build_kerala_chenda,
        master_volume=0.85,
    ),
    PackSpec(
        slug="resonant-drum",
        name="Resonant Drum",
        summary="Deep pitch-bending drum tap.",
        tags=["percussion", "classical", "drum"],
        builder=build_carnatic_mridangam,
        master_volume=0.85,
    ),
    PackSpec(
        slug="kalimba",
        name="Kalimba",
        summary="Plucked metal tines.",
        tags=["kalimba", "metal", "tines"],
        builder=build_kalimba_tines,
        master_volume=0.85,
    ),
    PackSpec(
        slug="pizzicato",
        name="Pizzicato",
        summary="Short finger-plucked string.",
        tags=["strings", "pizzicato", "acoustic"],
        builder=build_orchestral_pizzicato,
        master_volume=0.85,
    ),
    PackSpec(
        slug="soft-pudding-synth",
        name="Soft Pudding (Synth)",
        summary="Procedurally recreated soft dampened tap from v4.1.8.",
        tags=["custom", "procedural", "restored", "soft"],
        builder=build_soft_pudding_synth,
        master_volume=0.85,
    ),
    PackSpec(
        slug="muted-marshmallow",
        name="Muted Marshmallow",
        summary="Ultra-dead, heavy sub-bass thud. Extremely soft.",
        tags=["soft", "custom", "procedural", "thud"],
        builder=build_muted_marshmallow,
        master_volume=0.85,
    ),
    PackSpec(
        slug="felted-thock",
        name="Felted Thock",
        summary="Classic dampened enthusiast keyboard sound.",
        tags=["soft", "custom", "procedural", "thock"],
        builder=build_felted_thock,
        master_volume=0.85,
    ),
    PackSpec(
        slug="membrane-squish",
        name="Membrane Squish",
        summary="Retro rubber dome office keyboard feel.",
        tags=["soft", "custom", "procedural", "retro"],
        builder=build_membrane_squish,
        master_volume=0.85,
    ),
    PackSpec(
        slug="cork-tap",
        name="Cork Tap",
        summary="Earthy, dry tap on soft wood or cork.",
        tags=["soft", "custom", "procedural", "organic"],
        builder=build_cork_tap,
        master_volume=0.85,
    ),
    PackSpec(
        slug="velvet-whisper",
        name="Velvet Whisper",
        summary="Airy, ASMR-style quiet tap with smooth release.",
        tags=["soft", "custom", "procedural", "asmr"],
        builder=build_velvet_whisper,
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

        packs_dir = Path("packs") / spec.slug
        if packs_dir.exists():
            shutil.rmtree(packs_dir)
        shutil.copytree(pack_dir, packs_dir)

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

    shutil.copy2(index_path, Path("index.json"))
    print(f"Done. Index written to: {index_path} and root index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

