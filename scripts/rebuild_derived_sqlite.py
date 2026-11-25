#!/usr/bin/env python3
"""Rebuild the derived tables in data/tsa.db directly in SQLite.

Why this exists: the original pipeline is compute_*.py -> Postgres ->
pg_to_sqlite.py, but the VM's apt-installed PostgreSQL does not survive
VM replacements (pg_ctl is gone again as of 2026-11-24) and the Postgres
copy of atp_matches is stale anyway. The merged data lives in SQLite,
so the derived tables are rebuilt here, in place, from atp_matches,
using the SAME logic as the compute scripts (same walk order, same
K=32, same parse_score). Nothing in /report reads the form tables,
so atp_form_history / atp_games_form_history are NOT rebuilt here
(they stay at the old 199,389-row snapshot; rebuild them separately
if a consumer needs them).

Rebuilt tables:
- atp_elo_history, atp_elo_current   (compute_elo.py)
- atp_surface_elo_history            (compute_surface_elo.py)
- atp_match_games                    (compute_games.py)

Safety: rebuilds into a COPY of the DB, verifies row counts and
spot-checks (Tour Finals 2026 present, Sinner/Alcaraz current Elo
sane), and only then swaps the copy over data/tsa.db. The pre-merge
backup data/tsa.db.bak-20261124 is left untouched.

Usage (repo root, project venv):
    .venv/bin/python scripts/rebuild_derived_sqlite.py
"""

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo import DEFAULT_K, INITIAL_ELO, update_elo  # noqa: E402
from score_parser import parse_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "tsa.db"
WORK_PATH = REPO_ROOT / "data" / "tsa.db.rebuild"


def rebuild(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT tourney_id, match_num, tourney_date, winner_id, loser_id, "
        "surface, score FROM atp_matches "
        "WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL "
        "ORDER BY tourney_date, tourney_id, match_num"
    )
    matches = cur.fetchall()
    print(f"Matches to process: {len(matches)}")

    ratings = {}
    matches_played = {}
    last_date = {}
    elo_history = []
    overall = {}
    surface_rating = {}
    surface_n = {}
    surface_rows = []
    games_rows = []
    missing_surface = 0
    complete = incomplete = failed = 0

    for (tid, mnum, tdate, wid, lid, surface, score) in matches:
        # --- overall Elo (compute_elo.py) ---
        w = ratings.get(wid, INITIAL_ELO)
        l = ratings.get(lid, INITIAL_ELO)
        w_new, l_new, expected = update_elo(w, l, k=DEFAULT_K)
        ratings[wid] = w_new
        ratings[lid] = l_new
        matches_played[wid] = matches_played.get(wid, 0) + 1
        matches_played[lid] = matches_played.get(lid, 0) + 1
        last_date[wid] = tdate
        last_date[lid] = tdate
        elo_history.append(
            (tid, mnum, tdate, wid, lid, w, l, w_new, l_new, expected))

        # --- surface Elo (compute_surface_elo.py) ---
        w_over = overall.get(wid, INITIAL_ELO)
        l_over = overall.get(lid, INITIAL_ELO)
        w_over_new, l_over_new, _ = update_elo(w_over, l_over, k=DEFAULT_K)
        overall[wid] = w_over_new
        overall[lid] = l_over_new
        if surface is None:
            missing_surface += 1
            surface_rows.append(
                (tid, mnum, tdate, wid, lid, None,
                 w_over, l_over, None, None, 0, 0, None, None))
        else:
            w_key = (wid, surface)
            l_key = (lid, surface)
            w_surf = surface_rating.get(w_key, INITIAL_ELO)
            l_surf = surface_rating.get(l_key, INITIAL_ELO)
            w_n = surface_n.get(w_key, 0)
            l_n = surface_n.get(l_key, 0)
            w_surf_new, l_surf_new, _ = update_elo(w_surf, l_surf,
                                                   k=DEFAULT_K)
            surface_rating[w_key] = w_surf_new
            surface_rating[l_key] = l_surf_new
            surface_n[w_key] = w_n + 1
            surface_n[l_key] = l_n + 1
            surface_rows.append(
                (tid, mnum, tdate, wid, lid, surface,
                 w_over, l_over, w_surf, l_surf, w_n, l_n,
                 w_surf_new, l_surf_new))

        # --- match games (compute_games.py) ---
        wg, lg, is_complete = parse_score(score)
        if wg is None:
            failed += 1
            pct = None
        else:
            pct = (wg / (wg + lg)) if is_complete and (wg + lg) > 0 else None
            if is_complete:
                complete += 1
            else:
                incomplete += 1
        games_rows.append((tid, mnum, wg, lg, pct, 1 if is_complete else 0))

    # --- write (drop + recreate, same schema as the SQLite export) ---
    cur.execute("DROP TABLE IF EXISTS atp_elo_history")
    cur.execute("DROP TABLE IF EXISTS atp_elo_current")
    cur.execute("DROP TABLE IF EXISTS atp_surface_elo_history")
    cur.execute("DROP TABLE IF EXISTS atp_match_games")
    cur.execute(
        'CREATE TABLE "atp_elo_history" ("tourney_id" TEXT, "match_num" INTEGER, '
        '"tourney_date" DATE, "winner_id" INTEGER, "loser_id" INTEGER, '
        '"winner_elo_before" REAL, "loser_elo_before" REAL, '
        '"winner_elo_after" REAL, "loser_elo_after" REAL, '
        '"expected_winner_prob" REAL, PRIMARY KEY ("tourney_id", "match_num"))')
    cur.execute(
        'CREATE TABLE "atp_elo_current" ("player_id" INTEGER, "elo" REAL, '
        '"matches_played" INTEGER, "last_match_date" DATE, '
        'PRIMARY KEY ("player_id"))')
    cur.execute(
        'CREATE TABLE "atp_surface_elo_history" ("tourney_id" TEXT, "match_num" INTEGER, '
        '"tourney_date" DATE, "winner_id" INTEGER, "loser_id" INTEGER, "surface" TEXT, '
        '"winner_overall_before" REAL, "loser_overall_before" REAL, '
        '"winner_surface_before" REAL, "loser_surface_before" REAL, '
        '"winner_surface_n_before" INTEGER, "loser_surface_n_before" INTEGER, '
        '"winner_surface_after" REAL, "loser_surface_after" REAL, '
        'PRIMARY KEY ("tourney_id", "match_num"))')
    cur.execute(
        'CREATE TABLE "atp_match_games" ("tourney_id" TEXT, "match_num" INTEGER, '
        '"winner_games" INTEGER, "loser_games" INTEGER, "winner_games_pct" REAL, '
        '"is_complete" INTEGER, PRIMARY KEY ("tourney_id", "match_num"))')
    for i in range(0, len(elo_history), 5000):
        cur.executemany(
            "INSERT INTO atp_elo_history VALUES (?,?,?,?,?,?,?,?,?,?)",
            elo_history[i:i + 5000])
    cur.executemany(
        "INSERT INTO atp_elo_current VALUES (?,?,?,?)",
        [(pid, ratings[pid], matches_played[pid], last_date[pid])
         for pid in ratings])
    for i in range(0, len(surface_rows), 5000):
        cur.executemany(
            "INSERT INTO atp_surface_elo_history VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            surface_rows[i:i + 5000])
    for i in range(0, len(games_rows), 5000):
        cur.executemany(
            "INSERT INTO atp_match_games VALUES (?,?,?,?,?,?)",
            games_rows[i:i + 5000])
    conn.commit()
    print(f"elo_history rows: {len(elo_history)}; "
          f"players rated: {len(ratings)}")
    print(f"surface rows: {len(surface_rows)}; "
          f"missing surface: {missing_surface}")
    print(f"games: complete={complete} incomplete={incomplete} "
          f"failed={failed}")


def verify(conn):
    cur = conn.cursor()
    checks = {}
    for t in ("atp_elo_history", "atp_elo_current",
              "atp_surface_elo_history", "atp_match_games"):
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        checks[t] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM atp_matches")
    n_matches = cur.fetchone()[0]
    assert checks["atp_elo_history"] == n_matches, checks
    assert checks["atp_surface_elo_history"] == n_matches, checks
    assert checks["atp_match_games"] == n_matches, checks
    # Tour Finals 2026 (tid 2026-0605) must be present in all four.
    for t in ("atp_elo_history", "atp_surface_elo_history",
              "atp_match_games"):
        cur.execute(
            f"SELECT COUNT(*) FROM {t} WHERE tourney_id = '2026-0605'")
        assert cur.fetchone()[0] == 15, (t, "Tour Finals 2026 rows")
    # Current Elo sanity: Jannik Sinner and Carlos Alcaraz by full name
    # (other players share the surnames).
    cur.execute(
        "SELECT p.first_name, p.last_name, ROUND(c.elo, 1), "
        "c.matches_played, c.last_match_date FROM atp_elo_current c "
        "JOIN atp_players p ON p.player_id = c.player_id "
        "WHERE (p.first_name, p.last_name) IN "
        "(('Jannik', 'Sinner'), ('Carlos', 'Alcaraz')) "
        "ORDER BY c.elo DESC")
    top = cur.fetchall()
    print("Sinner/Alcaraz current Elo:", top)
    assert len(top) == 2
    assert all(e > 1800 for _, _, e, _, _ in top), top
    print("verify OK:", checks)
    return checks


def main():
    assert DB_PATH.exists(), DB_PATH
    if WORK_PATH.exists():
        WORK_PATH.unlink()
    shutil.copy2(DB_PATH, WORK_PATH)
    conn = sqlite3.connect(WORK_PATH)
    try:
        rebuild(conn)
        verify(conn)
    finally:
        conn.close()
    # Swap in: keep the pre-rebuild DB as .bak-rebuild (the pre-merge
    # backup data/tsa.db.bak-20261124 already exists separately).
    bak = REPO_ROOT / "data" / "tsa.db.bak-rebuild-20261124"
    if bak.exists():
        bak.unlink()
    DB_PATH.rename(bak)
    WORK_PATH.rename(DB_PATH)
    print(f"swapped in rebuilt DB; pre-rebuild DB kept at {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
