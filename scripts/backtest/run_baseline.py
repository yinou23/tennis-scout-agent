"""Run the locked-model baseline walk-forward backtest.

Usage (from the repo root, with the project venv):

    .venv/bin/python -m scripts.backtest.run_baseline
    .venv/bin/python -m scripts.backtest.run_baseline --odds-dir scripts/backtest/data
    .venv/bin/python -m scripts.backtest.run_baseline --odds-source valuebetennis

What it does, in order:
1. Loads atp_matches (1967-12-28 to 2024-12-31) in
   (tourney_date, tourney_id, match_num) order.
2. Verifies the in-memory Elo recompute against the stored
   atp_elo_history / atp_surface_elo_history tables (same
   sequential process, K=32). Stops if the max difference
   exceeds 1e-6.
3. Burn-in 1968-2000: Elo updates only, no scoring.
4. Scores 2001-2024: for each match, predict with pre-match
   Elo (locked model), record the outcome metrics, then update
   Elo with the result. No test set: --end-year >= 2025 is
   refused outright.
5. If --odds-dir holds odds files, aligns them (see align.py)
   and additionally scores the model against de-vigged closing
   prices per bookmaker (paired Brier difference, SE, z).

Outputs (all under scripts/backtest/outputs/, git-ignored):
    baseline_by_year.csv     one row per year, outcome + market metrics
    baseline_summary.json    pooled train (2001-2022) and validate
                             (2023-2024) metrics, run manifest
    baseline_predictions.csv one row per scored match (model prob,
                             market prob where matched)
    alignment_report.json    per-year, per-level match rates and
                             unmatched reasons (only when odds given)

Model interface for later work: the scorer calls
``elo_state.EloState.predict`` / ``.update`` only. A future model
(grid search, WElo, Glicko-2, features) plugs in by replacing
the EloState with an object exposing the same two methods plus
a ``name`` attribute; nothing else in this file changes.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from align import MAIN_TOUR_LEVELS, align_matches, match_rate_report  # noqa: E402
from db import get_connection, get_sqlite_path  # noqa: E402
from elo_state import EloState, verify_against_history  # noqa: E402
from metrics import devig, ece, paired_diff, score_outcome  # noqa: E402
from names import name_key  # noqa: E402
from odds import OddsRow, read_tennis_data_xlsx, read_valuebetennis_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "outputs"
TRAIN_END = 2022
VALIDATE_END = 2024


def _split_for(year):
    if year <= TRAIN_END:
        return "train"
    if year <= VALIDATE_END:
        return "validate"
    return "test"


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_matches(conn, end_year):
    cur = conn.cursor()
    cur.execute(
        "SELECT tourney_id, match_num, tourney_date, tourney_name, surface, "
        "round, tourney_level, winner_id, loser_id, winner_name, loser_name "
        "FROM atp_matches WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
        "AND tourney_date <= ? ORDER BY tourney_date, tourney_id, match_num",
        (f"{end_year}-12-31",),
    )
    cols = [d[0] for d in cur.description]
    matches = []
    for row in cur.fetchall():
        m = dict(zip(cols, row))
        m["tourney_date"] = _dt.date.fromisoformat(str(m["tourney_date"]))
        matches.append(m)
    return matches


def _load_odds(odds_dir, source, years):
    """Load odds rows per year. Returns (rows_by_year, manifest).

    Each year's pool also includes the adjacent years' files:
    season-openers can be played on 31 December of the previous
    year and the alignment date window reaches 2 days before the
    Sackmann tourney_date, so a January match can sit in the
    previous year's file. Rows keep their own file year in
    OddsRow.year; the pool key is the Sackmann match year.
    """
    rows_by_year, manifest = {}, []
    if odds_dir is None:
        return rows_by_year, manifest
    odds_dir = Path(odds_dir)
    seen = set()
    for year in years:
        rows = []
        for file_year in (year - 1, year, year + 1):
            td = odds_dir / f"{file_year}.xlsx"
            vb = odds_dir / f"valuebetennis-matchs-{file_year}.csv"
            if source in ("tennis-data", "auto") and td.exists():
                new_rows = read_tennis_data_xlsx(td, file_year)
                rows.extend(new_rows)
                if file_year not in seen:
                    seen.add(file_year)
                    manifest.append({"year": file_year, "source": "tennis-data.co.uk",
                                     "file": str(td), "sha256": _file_hash(td), "rows": len(new_rows)})
            elif source in ("valuebetennis", "auto") and vb.exists():
                new_rows = read_valuebetennis_csv(vb, file_year)
                rows.extend(new_rows)
                if file_year not in seen:
                    seen.add(file_year)
                    manifest.append({"year": file_year, "source": "valuebetennis",
                                     "file": str(vb), "sha256": _file_hash(vb), "rows": len(new_rows)})
        if rows:
            rows_by_year[year] = rows
    return rows_by_year, manifest


def _aggregate(records):
    """Aggregate one list of per-match records into metrics."""
    if not records:
        return None
    p_win = [r["p_model_winner"] for r in records]
    out = score_outcome(p_win)
    p_fav = [max(r["p_model_a"], 1.0 - r["p_model_a"]) for r in records]
    fav_won = [1 if (r["p_model_a"] >= 0.5) == (r["winner_is_a"]) else 0 for r in records]
    out["ece"] = ece(p_fav, fav_won)["ece"]
    out["favorite_win_rate"] = sum(fav_won) / len(fav_won)
    books = sorted({b for r in records for b in r.get("market", {})})
    out["market"] = {}
    for book in books:
        pairs = [(r["p_model_winner"], r["market"][book]) for r in records if book in r.get("market", {})]
        if not pairs:
            continue
        m_brier = sum((1.0 - p) ** 2 for _m, p in pairs) / len(pairs)
        m_ll = sum(-math.log(min(max(p, 1e-9), 1 - 1e-9)) for _m, p in pairs) / len(pairs)
        out["market"][book] = {
            "n_matched": len(pairs),
            "brier": m_brier,
            "log_loss": m_ll,
            **paired_diff([m for m, _p in pairs], [p for _m, p in pairs]),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="TSA locked-model walk-forward baseline")
    ap.add_argument("--db", default=None, help="SQLite path (default: data/tsa.db)")
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--odds-dir", default=None)
    ap.add_argument("--odds-source", choices=["auto", "tennis-data", "valuebetennis"], default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-elo-verify", action="store_true")
    args = ap.parse_args(argv)

    if args.end_year >= 2025 or args.start_year >= 2025:
        print("ERROR: test set 2025-2026 is locked. This run refuses to score it.", file=sys.stderr)
        return 2
    if args.start_year < 2001:
        print("ERROR: scoring starts at 2001 (odds coverage). Use burn-in for earlier years.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    if args.db:
        os.environ["SQLITE_PATH"] = args.db
    conn = get_connection()
    db_path = str(get_sqlite_path())

    if not args.skip_elo_verify:
        print("Verifying Elo recompute against atp_elo_history / atp_surface_elo_history ...")
        check = verify_against_history(conn, upto_date=f"{args.end_year}-12-31")
        print(f"  checked={check['checked']} max_overall_diff={check['max_overall_diff']:.2e} "
              f"max_surface_diff={check['max_surface_diff']:.2e} mismatches={check['mismatches']}")
        if check["error"] or check["max_overall_diff"] > 1e-6 or check["max_surface_diff"] > 1e-6:
            print(f"ERROR: Elo verification failed: {check['error']}", file=sys.stderr)
            return 1

    matches = _load_matches(conn, args.end_year)
    print(f"Matches loaded (to {args.end_year}): {len(matches)}")

    years = sorted({m["tourney_date"].year for m in matches if m["tourney_date"].year >= args.start_year})
    odds_by_year, odds_manifest = _load_odds(args.odds_dir, args.odds_source, years)
    print(f"Odds years loaded: {sorted(odds_by_year)} (source={args.odds_source})")

    state = EloState()
    records, alignment_reports = [], {}
    for m in matches:
        year = m["tourney_date"].year
        if year < args.start_year:
            state.update(m["winner_id"], m["loser_id"], m["surface"])
            continue
        # Predict BEFORE update: no leakage by construction.
        p_a, _p_blended, _ba, _bb = state.predict(m["winner_id"], m["loser_id"], m["surface"])
        # predict() is symmetric in the two ids: p_a here is the
        # probability the eventual winner wins. Record winner-side.
        rec = {
            "tourney_id": m["tourney_id"], "match_num": m["match_num"],
            "date": str(m["tourney_date"]), "tournament": m["tourney_name"],
            "surface": m["surface"], "round": m["round"],
            "winner_name": m["winner_name"], "loser_name": m["loser_name"],
            "year": year, "split": _split_for(year),
            "p_model_a": p_a, "p_model_winner": p_a, "winner_is_a": True,
            "market": {},
        }
        records.append(rec)
        state.update(m["winner_id"], m["loser_id"], m["surface"])

    # Alignment + market probabilities, per year (does not touch Elo).
    for year in years:
        rows = odds_by_year.get(year)
        if not rows:
            continue
        year_matches = [m for m in matches if m["tourney_date"].year == year]
        matched, unmatched = align_matches(year_matches, rows)
        alignment_reports[str(year)] = match_rate_report(year_matches, matched, unmatched)
        by_key = {(m["tourney_id"], m["match_num"]): o for m, o in matched}
        rec_by_key = {(r["tourney_id"], r["match_num"]): r for r in records if r["year"] == year}
        for key, odds_row in by_key.items():
            rec = rec_by_key.get(key)
            if rec is None:
                continue
            for book, (oa, ob) in odds_row.odds.items():
                try:
                    pa, pb = devig(oa, ob)
                except ValueError:
                    continue
                # Odds order is player_a/player_b as listed in the file.
                if name_key(odds_row.player_a) == name_key(rec["winner_name"]):
                    rec["market"][book] = pa
                elif name_key(odds_row.player_b) == name_key(rec["winner_name"]):
                    rec["market"][book] = pb
                else:
                    # Pair matched on (surname, initial) but the full
                    # name key differs: do not guess the orientation.
                    continue
        print(f"  {year}: odds rows={len(rows)} main-tour match rate="
              f"{alignment_reports[str(year)]['main_tour']['rate']} "
              f"(excl team/exhibition "
              f"{alignment_reports[str(year)]['main_tour_excl_team_exhibition']['rate']})")

    # Per-year + pooled aggregates.
    by_year = {}
    for year in years:
        recs = [r for r in records if r["year"] == year]
        agg = _aggregate(recs)
        agg["year"] = year
        agg["split"] = _split_for(year)
        by_year[year] = agg
    pooled = {}
    for split in ("train", "validate"):
        recs = [r for r in records if r["split"] == split]
        pooled[split] = _aggregate(recs)

    with open(out_dir / "baseline_by_year.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "split", "n", "brier", "log_loss", "accuracy",
                    "favorite_win_rate", "ece", "market_book", "n_matched",
                    "market_brier", "market_log_loss", "D", "se", "z"])
        for year in years:
            agg = by_year[year]
            if agg["market"]:
                for book, mb in agg["market"].items():
                    w.writerow([year, agg["split"], agg["n"], f"{agg['brier']:.4f}",
                                f"{agg['log_loss']:.4f}", f"{agg['accuracy']:.4f}",
                                f"{agg['favorite_win_rate']:.4f}", f"{agg['ece']:.4f}",
                                book, mb["n_matched"], f"{mb['brier']:.4f}",
                                f"{mb['log_loss']:.4f}", f"{mb['D']:.4f}",
                                f"{mb['se']:.4f}", f"{mb['z']:.2f}"])
            else:
                w.writerow([year, agg["split"], agg["n"], f"{agg['brier']:.4f}",
                            f"{agg['log_loss']:.4f}", f"{agg['accuracy']:.4f}",
                            f"{agg['favorite_win_rate']:.4f}", f"{agg['ece']:.4f}",
                            "", "", "", "", "", "", ""])

    with open(out_dir / "baseline_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tournament", "surface", "round", "winner_name", "loser_name",
                    "year", "split", "p_model_winner"] +
                   [f"p_market_{b}_winner" for b in sorted({b for r in records for b in r["market"]})])
        books = sorted({b for r in records for b in r["market"]})
        for r in records:
            w.writerow([r["date"], r["tournament"], r["surface"], r["round"],
                        r["winner_name"], r["loser_name"], r["year"], r["split"],
                        f"{r['p_model_winner']:.4f}"] +
                       [f"{r['market'][b]:.4f}" if b in r["market"] else "" for b in books])

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        head = None
    summary = {
        "model": "K32 overall Elo + surface blend prior100 + calibration a=0.8, b=0.0",
        "run_date": _dt.date.today().isoformat(),
        "db": db_path,
        "git_head": head,
        "splits": {"train": "2001-2022", "validate": "2023-2024", "test": "2025-2026 (LOCKED, not run)"},
        "odds_manifest": odds_manifest,
        "by_year": by_year,
        "pooled": pooled,
    }
    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "alignment_report.json", "w") as f:
        json.dump(alignment_reports, f, indent=2, default=str)

    for split in ("train", "validate"):
        agg = pooled[split]
        print(f"{split}: n={agg['n']} brier={agg['brier']:.4f} log_loss={agg['log_loss']:.4f} "
              f"accuracy={agg['accuracy']:.4f} ece={agg['ece']:.4f}")
        for book, mb in agg["market"].items():
            print(f"  vs {book}: n_matched={mb['n_matched']} market_brier={mb['brier']:.4f} "
                  f"D={mb['D']:.4f} se={mb['se']:.4f} z={mb['z']:.2f}")
    print(f"Wrote {out_dir}/baseline_by_year.csv, baseline_summary.json, "
          f"baseline_predictions.csv, alignment_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
