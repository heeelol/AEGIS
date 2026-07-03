"""
Phase 3 reach-run scorer
==========================
Diffs the ACTUAL sequence logged by ``phase3_reach_logger.py`` against the
INTENDED target sequence you performed (you know this order — you're the one
who executed the scripted reaches), and prints a row matching the "Hand pose"
test sheet's columns.

Best fit for the sequential single/one-hand-at-a-time blocks where each reach
has one specific intended bin (Block 1, Block 2, and most of Block 5's
gap/near-miss/side-angle rows). For alternating (Block 3) and simultaneous
(Block 4) reaches, the raw ``reach_log.csv`` + saved snapshots are still
there for manual scoring — those interaction patterns don't reduce to a
simple one-bin-per-reach comparison.

Run (from aegis-v2/):
    # RH-1: all 9 bins, 10 reaches each, in slot order 1..9
    python tools/score_phase3_run.py --dir tools/Dev_test_data/phase3_run_01 \\
        --bins 1-9 --repeats 10 --run RH-1 --block "Block 1 Right hand only" --hand right

    # RH-2: bottom 3 bins only (slots 7-9), 10 reaches each
    python tools/score_phase3_run.py --dir tools/Dev_test_data/phase3_run_02 \\
        --bins 7-9 --repeats 10 --run RH-2 --hand right

    # explicit custom order instead of a range
    python tools/score_phase3_run.py --dir tools/Dev_test_data/phase3_run_03 \\
        --targets 1x5,2x5,3x5,4x5,5x5,6x5,7x5,8x5,9x5 --run RH-3 --hand right
"""
from __future__ import annotations

import argparse
import csv
import difflib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("aegis.tools.score_phase3_run")


def slot_to_bin_id(n: int) -> str:
    """1..6 -> top row bin_0_0..bin_0_5, 7..9 -> bottom row bin_1_0..bin_1_2."""
    if 1 <= n <= 6:
        return f"bin_0_{n - 1}"
    if 7 <= n <= 9:
        return f"bin_1_{n - 7}"
    raise ValueError(f"Slot number must be 1..9, got {n}")


def parse_targets_spec(spec: str) -> list:
    """'1x10,2x10,9' -> [bin_0_0]*10 + [bin_0_1]*10 + [bin_1_2]."""
    targets = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" in token:
            slot_str, repeat_str = token.split("x", 1)
            slot, repeat = int(slot_str), int(repeat_str)
        else:
            slot, repeat = int(token), 1
        targets.extend([slot_to_bin_id(slot)] * repeat)
    return targets


def parse_bins_range(bins_range: str, repeats: int) -> list:
    """'1-9' + repeats=10 -> slots 1..9 each repeated 10x, in order."""
    start_str, end_str = bins_range.split("-", 1)
    start, end = int(start_str), int(end_str)
    targets = []
    for slot in range(start, end + 1):
        targets.extend([slot_to_bin_id(slot)] * repeats)
    return targets


def load_actual_sequence(log_path: Path, hand_filter: str) -> list:
    if not log_path.exists():
        raise FileNotFoundError(f"No reach_log.csv in {log_path.parent}")
    with open(log_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["index"]))
    sequence = []
    for r in rows:
        if not r["bin_id"]:
            continue  # exit event, not an entry
        if hand_filter != "any" and r["handedness"] != hand_filter:
            continue
        sequence.append(r["bin_id"])
    return sequence


def _target_blocks(intended: list) -> list:
    """Group intended into (bin_id, start, end) runs of consecutive identical targets."""
    blocks = []
    i = 0
    n = len(intended)
    while i < n:
        j = i
        while j < n and intended[j] == intended[i]:
            j += 1
        blocks.append((intended[i], i, j))
        i = j
    return blocks


def score_aligned(intended: list, actual: list) -> dict:
    """Sequence-alignment scoring (default): robust to spurious EXTRA entries in
    ``actual`` (e.g. a hand passing through a bottom bin's geofence on its way up
    to a top bin logs a real-but-unintended transit "entry"). A naive positional
    diff treats every such insertion as corrupting all subsequent comparisons —
    this instead finds the best-matching subsequence (LCS via difflib), so one
    stray mid-run bin doesn't cascade into false mismatches for the rest of the
    run. Reports per-target-block matched/total instead of reach-by-reach pairs,
    since insertions break the 1:1 reach-index correspondence anyway.
    """
    sm = difflib.SequenceMatcher(None, intended, actual, autojunk=False)
    matched_idx = set()
    for blk in sm.get_matching_blocks():
        matched_idx.update(range(blk.a, blk.a + blk.size))

    n = len(intended)
    correct = len(matched_idx)
    accuracy = (correct / n * 100.0) if n else 0.0

    per_target = [{"bin_id": bid, "matched": sum(1 for i in range(start, end) if i in matched_idx),
                   "total": end - start}
                  for bid, start, end in _target_blocks(intended)]
    return {"n": n, "correct": correct, "accuracy": accuracy, "per_target": per_target,
            "actual_len": len(actual)}


def format_failure_notes_aligned(result: dict) -> str:
    problems = [t for t in result["per_target"] if t["matched"] < t["total"]]
    if not problems:
        return ""
    parts = [f"{t['bin_id']}: {t['matched']}/{t['total']} matched" for t in problems]
    return "; ".join(parts)


def score_strict(intended: list, actual: list) -> dict:
    """Naive positional diff (legacy / --strict). Only meaningful when ``actual``
    has no spurious extra entries — a single stray insertion shifts every
    comparison after it, so prefer ``score_aligned`` for real hand-tracking logs.
    """
    n = len(intended)
    correct = 0
    mismatches = []
    for i in range(n):
        a = actual[i] if i < len(actual) else None
        if a == intended[i]:
            correct += 1
        else:
            mismatches.append((i + 1, intended[i], a))
    accuracy = (correct / n * 100.0) if n else 0.0
    return {"n": n, "correct": correct, "accuracy": accuracy, "mismatches": mismatches,
            "actual_len": len(actual)}


def format_failure_notes_strict(result: dict) -> str:
    if not result["mismatches"]:
        return ""
    parts = [f"reach #{i} intended {intended} got {actual or '(none)'}"
             for i, intended, actual in result["mismatches"][:10]]
    if len(result["mismatches"]) > 10:
        parts.append(f"... +{len(result['mismatches']) - 10} more")
    return "; ".join(parts)


def score_from_summary(summary_path: Path) -> dict:
    """Read target_summary.csv from the target-gated logger directly — no
    alignment guessing needed, since only the intended bin was ever registered
    with the assignment engine per step, so every logged sample is already
    unambiguous."""
    with open(summary_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    per_target = [{"bin_id": r["bin_id"], "matched": int(r["samples_confirmed"]),
                   "total": int(r["repeats_intended"])} for r in rows]
    n = sum(t["total"] for t in per_target)
    correct = sum(min(t["matched"], t["total"]) for t in per_target)
    accuracy = (correct / n * 100.0) if n else 0.0
    return {"n": n, "correct": correct, "accuracy": accuracy, "per_target": per_target, "actual_len": correct}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a Phase 3 reach run against an intended target sequence")
    ap.add_argument("--dir", required=True, help="phase3_run_<N> folder")
    ap.add_argument("--from-summary", action="store_true",
                     help="score target_summary.csv from phase3_reach_logger.py's target-gated capture "
                          "(no --targets/--bins needed — the sequence is already recorded)")
    ap.add_argument("--targets", default=None, help="(reach_log.csv mode) explicit sequence, e.g. '1x10,2x10,9x5'")
    ap.add_argument("--bins", default=None, help="(reach_log.csv mode) slot range shorthand, e.g. '1-9' or '7-9'")
    ap.add_argument("--repeats", type=int, default=10, help="(--bins only) reaches per target")
    ap.add_argument("--hand", default="any", choices=["any", "left", "right"],
                     help="(reach_log.csv mode) only count entries by this hand (default: any)")
    ap.add_argument("--pass-threshold", type=float, default=95.0, help="pass criterion, %% correct")
    ap.add_argument("--strict", action="store_true",
                     help="(reach_log.csv mode) use naive positional diff instead of sequence-alignment scoring "
                          "(only meaningful if the log has zero spurious transit entries)")
    ap.add_argument("--run", default="", help="Run # / ConditionID column")
    ap.add_argument("--block", default="", help="Block column")
    args = ap.parse_args()

    run_dir = Path(args.dir)

    if args.from_summary:
        summary_path = run_dir / "target_summary.csv"
        if not summary_path.exists():
            logger.error("No target_summary.csv in %s", run_dir)
            return
        result = score_from_summary(summary_path)
        failure_notes = format_failure_notes_aligned(result)  # same per_target shape
        passed = result["accuracy"] >= args.pass_threshold
    else:
        if bool(args.targets) == bool(args.bins):
            logger.error("Pass exactly one of --targets or --bins (or use --from-summary)")
            return
        intended = parse_targets_spec(args.targets) if args.targets else parse_bins_range(args.bins, args.repeats)
        if not intended:
            logger.error("Empty target sequence.")
            return
        try:
            actual = load_actual_sequence(run_dir / "reach_log.csv", args.hand)
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        if args.strict:
            result = score_strict(intended, actual)
            failure_notes = format_failure_notes_strict(result)
        else:
            result = score_aligned(intended, actual)
            failure_notes = format_failure_notes_aligned(result)
        passed = result["accuracy"] >= args.pass_threshold

        if result["actual_len"] != result["n"]:
            logger.info("Logged %d entr%s, intended sequence has %d (extra entries are expected — e.g. a hand "
                         "passing through a bottom bin's geofence en route to a top bin; alignment scoring absorbs these)",
                         result["actual_len"], "y" if result["actual_len"] == 1 else "ies", result["n"])

    print()
    print("Run\tBlock\tHand(s)\tReaches per target\tTotal reaches\tCorrect assignments\tCorrect (%)\tPass/Fail\tFailure notes\tEvidence ref")
    print(f"{args.run}\t{args.block}\t{args.hand}\t{args.repeats}\t{result['n']}\t"
          f"{result['correct']}\t{result['accuracy']:.0f}%\t{'Pass' if passed else 'Fail'}\t{failure_notes}\t{run_dir}")
    print()


if __name__ == "__main__":
    main()
