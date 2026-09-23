# mario_bugged — the variant that will carry deliberate bugs

Created as a byte-identical copy of `../mario_clean/`. **It currently contains
no intentional bug**: `INJECTED_BUGS.json` is empty, and
`tests/test_game_variants.py` fails if this game differs from mario_clean in
any file a declared bug does not name.

Adding a bug later (only when the project owner specifies one):

1. Edit files under `data/` (or `resources/`) here, never in mario_clean.
2. Add an entry to `INJECTED_BUGS.json`:
   `{"id": "short-name", "summary": "...", "files": ["data/components/mario.py"]}`
   listing every game file the bug touches (paths relative to this folder).
3. Run the tests: mario_clean must still match its pinned hash, and every
   difference here must be declared.

Run the dashboard on this variant with `python app.py --game mario_bugged`.
Every incident records the variant name and its game-tree SHA-256, so a
report always says which game produced it.

This file and INJECTED_BUGS.json are metadata, not part of the game tree hash.
