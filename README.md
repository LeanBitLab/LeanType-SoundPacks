# LeanType Sound Packs

Official sound packs catalog and repository tools for [LeanType Keyboard](https://github.com/LeanBitLab/LeanType).

## Overview

LeanType supports high-performance, memory-safe, data-only keypress sound packs. Sound packs are distributed as lightweight `.zip` archives containing audio files and a `pack.json` manifest.

## Available Packs in this Repository

| Pack Name | ID | Description |
|---|---|---|
| **Mechanical Thock** | `dev.leantype.sounds.mechanical_thock` | Deep, lubricated switch thock with heavy spacebar clack |
| **Kailh Box Jade Clicky** | `dev.leantype.sounds.box_jade_clicky` | Ultra-crisp high-pitched tactile click bar switches |
| **Vintage Royal Typewriter** | `dev.leantype.sounds.vintage_typewriter` | Cast-iron hammer strike with newline carriage chime on Enter |
| **8-Bit Retro Arcade** | `dev.leantype.sounds.arcade_8bit` | Nostalgic chiptune square-wave gaming blips and chirps |
| **Water Bubble / Pop** | `dev.leantype.sounds.bubble_pop` | Satisfying soft liquid bubble pop feedback |
| **Teak Woodblock Minimal** | `dev.leantype.sounds.woodblock_teak` | Natural acoustic wooden mallet resonance |

## Creating a Sound Pack

1. Create a folder in `packs/<pack_name>/`.
2. Add your audio files (`.ogg` or `.wav`) in `packs/<pack_name>/audio/`.
3. Create `packs/<pack_name>/pack.json`:
   ```json
   {
     "schemaVersion": 1,
     "id": "dev.leantype.sounds.my_pack",
     "name": "My Custom Sound Pack",
     "summary": "Short description of the acoustic profile",
     "versionCode": 1,
     "versionName": "1.0.0",
     "author": "Your Name",
     "license": "CC0-1.0",
     "defaultMasterVolume": 0.85,
     "preview": "audio/keypress_default_1.ogg",
     "sounds": {
       "keypress.default": {
         "files": ["audio/keypress_default_1.ogg", "audio/keypress_default_2.ogg"],
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
4. Validate and package your pack:
   ```bash
   python tools/validate_pack.py packs/my_pack
   python tools/package_pack.py packs/my_pack dist/my_pack.zip
   python tools/generate_index.py dist --out index.json
   ```

## License

All sound assets generated in this repository are dedicated to the public domain under **CC0 1.0 Universal**.
