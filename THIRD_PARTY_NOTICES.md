# Third-Party Notices

The `mario_clean/` and `mario_bugged/` directories are **not** original work of
this project and are **not** covered by the MIT license in `LICENSE`.

---

## mario_clean/ and mario_bugged/ — Super Mario Bros Level 1 (Python/Pygame)

Two copies of the same upstream game. `mario_clean/` is the unmodified
baseline; `mario_bugged/` exists so this project can add deliberate test bugs
for its QA agent to find, and differs from it only where such a bug is
declared in `mario_bugged/INJECTED_BUGS.json`. Neither changes the terms below.

**Original author:** Justin Meister
(credited in the source files as `__author__ = 'justinarmstrong'`)

**Upstream project:** https://github.com/justinmeister/Mario-Level-1

**License:** **None.** The upstream repository does not include a LICENSE
file or state any open-source license. Its README contains exactly one
statement about usage:

> "This project is intended for non-commercial educational purposes."

### What this means in practice

Because no license was granted, the default of copyright law applies: all
rights are reserved by the original author. The non-commercial educational
statement above is the only permission the author has expressed.

**Therefore:**

- Use this project for **learning, coursework, and personal experimentation** —
  that is squarely within what the original author described.
- **Do not sell it, put it in a product, or use it commercially** in any form.
- If you want to redistribute or publish this repository beyond
  personal/educational sharing, ask the original author for permission
  first. A public GitHub repository is a form of redistribution.

---

## Game artwork and audio

The sprite sheets in `<variant>/resources/graphics/` and the audio in
`<variant>/resources/music/` and `<variant>/resources/sound/` (in both
`mario_clean/` and `mario_bugged/`) are assets
from **Super Mario Bros.**, which is the intellectual property of
**Nintendo Co., Ltd.**

The same graphics appear in this project's screenshots and recordings: the
images in `assets/` and the frames, GIFs and PDFs of every incident report.

These assets are **not licensed** by this project, by the upstream author, or
by anyone else here. They are included only because the original educational
project included them. Nintendo has not granted any permission for their use.

This project is not affiliated with, endorsed by, or sponsored by Nintendo.

---

## static/vendor/socket.io.min.js — Socket.IO client

**Version:** 4.7.2 · **Copyright:** (c) 2014-2023 Guillermo Rauch ·
**License:** MIT

Shipped in the repository (not fetched from a CDN) so the dashboard works
with no internet connection. The copyright and license notice is kept in the
file's own header comment, as the MIT license requires. Upstream:
https://github.com/socketio/socket.io

---

## Python dependencies

The runtime packages in `requirements.txt` and `pyproject.toml` are
installed from PyPI (PyTorch from its own index) at setup time and are **not
redistributed** as part of this repository. Each keeps its own license
(as declared in its package metadata):

| Package | Used for | License |
|---|---|---|
| PyTorch | the neural network | BSD-3-Clause |
| Stable-Baselines3 | PPO | MIT |
| Gymnasium | the environment API | MIT |
| Pygame | the game engine | LGPL (version 2.1, per its license file) |
| NumPy | arrays | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| cloudpickle | checkpoint serialisation | BSD-3-Clause |
| OpenCV (`opencv-python`) | frame stream, PNG evidence | Apache-2.0 |
| Pillow | incident GIFs | MIT-CMU |
| fpdf2 | PDF incident reports | LGPL-3.0-only |
| Flask | the dashboard server | BSD-3-Clause |
| Flask-SocketIO | live dashboard updates | MIT |

Pygame and fpdf2 are LGPL: they are used as ordinary, unmodified installed
libraries, which the LGPL permits for software under any license. The
development tools (pytest, pytest-cov, Ruff, mypy, pre-commit, uv) are not
part of the running project. Refer to each project for its exact terms.

---

## The final 16M brain (GitHub Release asset)

`glitch_hunter_final_brain_16M.zip`, installed by `tools/final_brain.py`, is
this project's own work (trained weights, coverage map, reachability mask
and closure record); it contains no third-party code or assets.
