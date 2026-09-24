# mario_bugged — the variant that carries deliberate bugs

Created as a byte-identical copy of `../mario_clean/`. Since 2026-09-24 it
carries **six deliberate benchmark bugs**, specified by the project owner and
declared in `INJECTED_BUGS.json` (location, trigger, clean vs. injected
behaviour, implementation, reproducibility, what Objective 3 detects):

| # | id | where (level x) | what goes wrong |
|---|---|---|---|
| 1 | `stair-clip` | first staircase, the two top blocks of both tall columns (5874-5914, 6001-6041) | no collision in those blocks: Mario sinks 86 px into a column and walks into it |
| 2 | `pipe-clip` | pipe 4 (2445-2528) | only its left rim is solid: Mario drops into the pipe from its top, walks into it from the right |
| 3 | `ceiling-clip` | lone brick at 5058-5101, y 365 | no collision while Mario moves up: a jump goes straight through it |
| 4 | `invisible-wall` | open ground after the bricks at 4330 (4412-4452, y 452-538) | an undrawn two-tile collider blocks Mario; he can stand on it |
| 5 | `false-goomba-hit` | goomba14, the first of the last Goomba pair (spawned at x 6800) | its hurt box is 36 px bigger on every side: Mario dies with a visible gap |
| 6 | `open-sky-jump` | open sky at the start (jump taken at 250-480) | take-off x2.2 and no early-release cut: Mario flies ~300 px above the screen |

Every changed line carries `INJECTED BUG <id>`. Nothing else differs from
mario_clean, and three checks in `tests/test_game_variants.py` enforce that:
every changed **file** is named by a declared bug, every changed **block of
lines** carries the marker of a declared bug that names that file, and the
**whole** clean→bugged diff hashes to the manifest's `diff_sha256` (the
implementation that was validated). `tests/test_injected_bugs.py` proves what
each bug does against mario_clean, with a control spot per bug.

Changing a bug (only when the project owner specifies it):

1. Edit files under `data/` here, never in mario_clean, and mark every changed
   line block with `# INJECTED BUG <id>`.
2. Update the bug's entry in `INJECTED_BUGS.json`.
3. Re-validate (`pytest tests/test_injected_bugs.py`, and check the agent can
   still reach it), then re-pin `diff_sha256`
   (`python -c "from reporting import variants; print(variants.bug_diff_sha256('mario_bugged'))"`).

Run the dashboard on this variant with `python app.py --game mario_bugged`,
or pick "Mario Game (Bugged)" in the dashboard. Every incident records the
variant name, its game-tree SHA-256 and the declared bug ids, so a report
always says which game produced it.

This file and INJECTED_BUGS.json are metadata, not part of the game tree hash.
