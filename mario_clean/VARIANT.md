# mario_clean — the CLEAN baseline game

The canonical Super Mario Bros Level 1-1 clone (vendored third-party code; see
THIRD_PARTY_NOTICES.md). Every trained brain, the coverage mask and the
Objective-2 results were produced on exactly this game.

**Never put an intentional bug here.** This variant is the trustworthy
reference that the bugged variant is compared against. Its game tree
(`data/` and `resources/`) is pinned by `config.CLEAN_GAME_TREE_SHA256`, and
`tests/test_game_variants.py` fails on any change to it.

Deliberate bugs belong in `../mario_bugged/`, declared in its
`INJECTED_BUGS.json`.

This file is metadata, not part of the game tree hash.
