# LeanType Sound Packs

Official sound packs repository and creator guide for [LeanType Keyboard](https://github.com/LeanBitLab/LeanType).

---

## 🎵 Available Official Packs

| Pack Name | ID | Description |
|---|---|---|
| **Deep Thock** | `dev.leantype.sounds.thock` | Deep lubricated switch clack |
| **Crisp Click** | `dev.leantype.sounds.clicky` | High-pitched sharp click |
| **Tactile Pop** | `dev.leantype.sounds.tactile` | Snappy tactile bump and pop |
| **Mechanical Click** | `dev.leantype.sounds.mechanical` | Retro mechanical spring click |
| **Typewriter** | `dev.leantype.sounds.typewriter` | Vintage carriage and chime |
| **Creamy Linear** | `dev.leantype.sounds.creamy` | Soft dampened linear tap |
| **8-Bit Chiptune** | `dev.leantype.sounds.chiptune` | Retro square-wave arcade blips |
| **Glass Marble** | `dev.leantype.sounds.glass` | Polished mineral tap |
| **Bubble Pop** | `dev.leantype.sounds.bubble` | Soft liquid droplet burst |
| **Woodblock** | `dev.leantype.sounds.woodblock` | Acoustic wooden mallet tap |
| **Piano** | `dev.leantype.sounds.piano` | Warm harmonic key strike |
| **Acoustic Pluck** | `dev.leantype.sounds.acoustic-pluck` | Plucked nylon string tone |
| **Folk Drum** | `dev.leantype.sounds.folk-drum` | High-tension rim and drum hit |
| **Resonant Drum** | `dev.leantype.sounds.resonant-drum` | Deep pitch-bending drum tap |
| **Kalimba** | `dev.leantype.sounds.kalimba` | Plucked metal tines |
| **Pizzicato** | `dev.leantype.sounds.pizzicato` | Short finger-plucked string |

---

## 🛠️ User Guide: How to Create Your Own Custom Sound Pack

Creating and importing your own custom sound pack into LeanType is fast and simple.

### Method 1: Create a Full Sound Pack (.zip)

#### 1. Copy the Starter Template
Copy the template folder `templates/starter-pack` to a new folder:
```bash
cp -r templates/starter-pack packs/my_sound_pack
```

#### 2. Add Your Audio Files
Place short `.ogg` or `.wav` audio files in `packs/my_sound_pack/audio/`:
```text
packs/my_sound_pack/
  ├── pack.json
  ├── license.txt
  └── audio/
      ├── keypress_default_1.ogg
      ├── keypress_default_2.ogg
      ├── space.ogg
      ├── delete.ogg
      └── return.ogg
```

#### 3. Edit `pack.json`
Open `pack.json` and configure your metadata and sound events:
```json
{
  "schemaVersion": 1,
  "id": "dev.leantype.sounds.my_pack",
  "name": "My Custom Keyboard Sounds",
  "summary": "Crisp tactile clicks with custom spacebar",
  "versionCode": 1,
  "versionName": "1.0.0",
  "author": "Your Name",
  "license": "CC0-1.0",
  "defaultMasterVolume": 0.85,
  "preview": "audio/keypress_default_1.ogg",
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
```

#### 4. Supported Playback Modes
- `"single"`: Plays the first audio file.
- `"random"`: Picks randomly among the listed audio files for natural variation.
- `"cycle"`: Plays the listed audio files in round-robin order.

#### 5. Supported Key Events & Fallbacks
- `keypress.default`: Used for standard letters and numbers.
- `keypress.space`: Spacebar key (falls back to `keypress.default` if omitted).
- `keypress.delete`: Backspace key (falls back to `keypress.default` if omitted).
- `keypress.return`: Enter / Return key (falls back to `keypress.default` if omitted).
- `keypress.shift`: Shift / Caps Lock key (falls back to `keypress.default` if omitted).
- `keypress.symbol`: Symbol / 123 switch key (falls back to `keypress.default` if omitted).

#### 6. Validate & Package
Validate your pack structure and create the `.zip`:
```bash
python tools/validate_pack.py packs/my_sound_pack
python tools/package_pack.py packs/my_sound_pack dist/my_sound_pack.zip
```

---

### Method 2: Import a Single Audio File (Quick Method)

If you only have a single `.ogg`, `.wav`, or `.mp3` audio clip (like a single click or pop):
1. Transfer the audio file to your phone.
2. In LeanType:
   - Open **Settings** → **Plugins & Capabilities** → **Keypress Audio & Sound Packs**.
   - Tap **Sound Style** → **Import .zip**.
   - Select your `.ogg`, `.wav`, or `.mp3` file.
3. LeanType will automatically package it as a custom profile with instant playback!

---

### Method 3: Procedural Python Synthesis (Algorithmic Modeling)

You can synthesize sound packs from scratch using pure mathematics, digital signal processing (DSP), and physical modeling in Python without needing physical audio recordings.

#### 1. Setup Dependencies
```bash
pip install numpy scipy soundfile
```

#### 2. Write Your Custom Sound Synthesis Builder
In [`tools/generate_all_soundpacks.py`](tools/generate_all_soundpacks.py), define a sound builder function using the built-in DSP primitives:
- `tone(duration, f0, f1, wave='sine'|'triangle'|'square')`: Generates frequency sweeps and waveforms.
- `decay_env(duration, tau, attack_s)`: Generates exponential acoustic decay envelopes.
- `_modal_layers(duration, base_freq, ratios, gains, taus)`: Additive acoustic resonance model.
- `_ks_pluck(rng, freq, duration, brightness)`: Karplus-Strong physical plucked-string model.
- `lowpass()`, `highpass()`, `bandpass()`: Digital Butterworth / SOS filtering.
- `finalize(audio, target_db=-2.0)`: Peak normalizes (-2 dBFS), removes DC bias, and applies micro anti-pop fades.

Example:
```python
def build_custom_clack(rng: np.random.Generator) -> Dict[str, Dict]:
    def hit(freq: float, duration: float) -> np.ndarray:
        body = tone(duration, freq, freq * 0.85) * decay_env(duration, 0.02)
        click = bandpass(noise(rng, 0.005), 1500.0, 4500.0) * decay_env(0.005, 0.002)
        return finalize(mix([(body, 0.8), (click, 0.4)]))

    return {
        "keypress.default": ev([hit(300.0, 0.06), hit(320.0, 0.06)], "random"),
        "keypress.space": ev([hit(150.0, 0.10)], "single"),
        "keypress.delete": ev([hit(500.0, 0.04)], "single"),
        "keypress.return": ev([hit(220.0, 0.08)], "single"),
    }
```

#### 3. Register and Run
Add your `PackSpec` to the `PACKS` list:
```python
PackSpec(
    slug="my-custom-sound",
    name="My Custom Sound",
    summary="Algorithmic mechanical clack.",
    tags=["custom", "procedural"],
    builder=build_custom_clack,
)
```

Generate the sound pack `.zip`, previews, and `index.json`:
```bash
python tools/generate_all_soundpacks.py --base-url https://raw.githubusercontent.com/YOUR_USER/LeanType-SoundPacks/main/dist
```

---

## 📲 How to Import into LeanType on Your Device

1. Transfer your `.zip` (or `.ogg` / `.wav` / `.mp3`) file to your phone (via USB, Downloads, Telegram, Google Drive, etc.).
2. Open **LeanType Settings**.
3. Go to **Plugins & Capabilities** → **Keypress Audio & Sound Packs**.
4. Tap **Sound Style** → **Import .zip**.
5. Select your `.zip` or audio file using the system file picker.
6. LeanType will validate, install, activate, and preview your sound pack immediately!

---

## 🎧 Audio Guidelines & Recommendations

- **Format**: OGG Vorbis / Opus or WAV (Mono, 44.1 kHz or 48 kHz).
- **Duration**: 20 ms to 150 ms (keep them short for ultra-low latency).
- **Attack**: Zero leading silence at the start of the audio file for instant trigger response.
- **Normalization**: Peak normalized around -3 dB to -1 dB.
- **File Size**: Under 500 KB per audio variant, under 2 MB total pack size.

---

## 📜 License

Sound assets generated by LeanType Sound Lab in this repository are dedicated to the public domain under [CC0 1.0 Universal](LICENSE).
