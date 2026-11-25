#!/usr/bin/env python3
"""Merge TennisMyLife 2026.csv into data/tsa.db (one-off, 2026-11-24).

Verified facts about the DB (checked before writing):
- DB 2026 coverage: tournaments starting on/before 2026-05-25 (AO, RG
  complete; 1,449 rows). Missing: everything from June on (Wimbledon,
  US Open, summer/fall, ATP Finals). Only RG straddles the boundary and
  it is already complete in the DB, so no split-tournament renumbering.
- tourney_id format is mixed in the DB: Slams unpadded ('2026-520',
  '2026-580'), 250/500/Masters zero-padded ('2026-0301'). Dedupe
  normalizes by stripping leading zeros. Inserted rows keep the DB's
  exact id when the tournament exists, else TML's id as-is.
- tourney_name: DB uses 'Us Open' (not 'US Open') and 'Tour Finals'
  (not 'ATP Finals'). Those two are normalized on insert.
- tourney_date in the DB is the ATP official start date (usually the
  Monday; IW is a Wednesday, Finals a Sunday). For new tournaments we
  use the Monday of the week of the first match, Sunday -> following
  Monday (matches the DB's Slam convention; 1-2 day deviation on
  Wed/Sun-start events is ordering-equivalent).
- match_num: DB and TML number differently (DB Slams start at 100,
  250s at ~359; TML starts at 1). Dedupe NEVER uses match_num. Within
  a tournament both order by round, so TML's match_num is inserted
  as-is (internally consistent per tournament, which is all the
  backtest ordering needs).
- Score notation differs for retirements: TML '6-3 RET', DB
  '6-3 0-0 RET'. Same match; dedupe does not use score.
- TML seed column sometimes holds 'Q' (qualifier); DB puts that in
  the entry column and leaves seed NULL. Handled on insert.
- TML player ids are alphanumeric, ignored. Players are resolved by
  name (normalization + prefix/subsequence + reversed-name rules),
  disambiguated by id already used in atp_matches 2025+, then IOC,
  then age vs dob, then hand != 'U', then lowest id. Truly new
  players are inserted into atp_players with new ids (max+1), dob NULL.

Usage:
  python3 scripts/merge_tml_2026.py --csv /tmp/tml2026_check.csv [--dry-run]
  Default DB: data/tsa.db. On write, backs up to data/tsa.db.bak-20261124.
"""
import argparse
import csv
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import date, timedelta

LEVEL_MAP = {"250": "A", "500": "A", "M": "M", "G": "G", "F": "F", "D": "D", "A": "A"}

COL_MAP = {
    "w_1stIn": "w_1st_in", "w_1stWon": "w_1st_won", "w_2ndWon": "w_2nd_won",
    "w_SvGms": "w_sv_gms", "w_bpSaved": "w_bp_saved", "w_bpFaced": "w_bp_faced",
    "l_1stIn": "l_1st_in", "l_1stWon": "l_1st_won", "l_2ndWon": "l_2nd_won",
    "l_SvGms": "l_sv_gms", "l_bpSaved": "l_bp_saved", "l_bpFaced": "l_bp_faced",
}

NAME_OVERRIDES = {
    "Aleksandr Shevchenko": "Alexander Shevchenko",
    "Abdullah Shelbayh": "Abedallah Shelbayh",
    "Yunchaokete Bu": "Bu Yunchaokete",
    "Soonwoo Kwon": "Soon Woo Kwon",
    "Tung-Lin Wu": "Tung Lin Wu",
    "Chun-Hsin Tseng": "Chun Hsin Tseng",
    "Kaan Isik Kosaner": "Kaan Isak Kosaner",
    "Fares Zakaria": "Faris Zakaryia",
    "Igor Marcondes": "Igor Ribeiro Marcondes",
}

# Verified DB errors to correct on write (not TML/DB disagreements to
# guess at: a Mexican born 1984 cannot play Davis Cup for Paraguay).
# The DB's PAR-ROU tie rows credit 'Daniel Vallejo' (118808, MEX, 1984);
# the player is Adolfo Daniel Vallejo (209226, PAR, 2004). Delete the two
# wrong rows so the correct TML versions insert without duplicating.
WRONG_ROWS = [
    ("2026-M-DC-2026-WG1-PO-PAR-ROU-01", 118808, 212244),  # Vallejo d. Ghetu
    ("2026-M-DC-2026-WG1-PO-PAR-ROU-01", 118808, 207502),  # Vallejo d. Palosi
]

# DB's existing names for these events (TML -> DB)
TOURNEY_NAME_MAP = {"US Open": "Us Open", "ATP Finals": "Tour Finals"}
TOURNEY_ID_MAP = {"2026-605": "2026-0605"}  # Finals, matches DB 2025-0605

MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age", "score", "best_of", "round",
    "minutes", "w_ace", "w_df", "w_svpt", "w_1st_in", "w_1st_won",
    "w_2nd_won", "w_sv_gms", "w_bp_saved", "w_bp_faced", "l_ace", "l_df",
    "l_svpt", "l_1st_in", "l_1st_won", "l_2nd_won", "l_sv_gms",
    "l_bp_saved", "l_bp_faced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]


def norm_name(n):
    # lowercase, hyphens/dots -> space, drop apostrophes, strip accents
    # (Nuñez -> Nunez), collapse whitespace.
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(
        n.lower().replace("-", " ").replace("'", "").replace(".", " ").split()
    )


def strip_tid(t):
    """Normalize a tourney id for comparison: strip leading zeros."""
    parts = t.split("-")
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}-{int(parts[1])}"
    return t


def norm_tie_id(t):
    """Normalize a Davis Cup tie id: sort the two team codes.

    TML and the DB sometimes list the same tie in opposite team order
    ('CAN-BRA' vs 'BRA-CAN'). Sorting the codes makes them match.
    """
    parts = t.split("-")
    if len(parts) >= 3 and "DC" in parts and parts[-1].isdigit():
        a, b = parts[-3], parts[-2]
        parts[-3], parts[-2] = sorted([a, b])
        return "-".join(parts)
    return t


def tourney_monday(d):
    """Monday of the week of d; a Sunday belongs to the following week."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d - timedelta(days=d.weekday())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/tml2026_check.csv")
    ap.add_argument("--db", default="data/tsa.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"TML rows: {len(rows)}")

    min_date = {}
    for r in rows:
        d = date.fromisoformat(r["tourney_date"])
        tid = r["tourney_id"]
        if tid not in min_date or d < min_date[tid]:
            min_date[tid] = d
    monday = {tid: tourney_monday(d) for tid, d in min_date.items()}

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    # existing DB dedupe keys, by resolved player id (not by name: the two
    # sources spell several names differently, e.g. Aleksandr/Alexander
    # Shevchenko, Yunchaokete Bu / Bu Yunchaokete).
    cur.execute(
        "SELECT tourney_id, tourney_level, winner_id, loser_id FROM atp_matches"
    )
    db_keys = set()
    db_d_keys = set()
    for tid, level, wid, lid in cur.fetchall():
        if level == "D":
            db_d_keys.add((norm_tie_id(tid), wid, lid))
        else:
            db_keys.add((strip_tid(tid), wid, lid))
    # existing tourney_id per stripped tid (to reuse the DB's exact id)
    cur.execute("SELECT DISTINCT tourney_id FROM atp_matches")
    db_tid_by_stripped = {}
    for (tid,) in cur.fetchall():
        db_tid_by_stripped.setdefault(strip_tid(tid), tid)

    cur.execute(
        "SELECT player_id, first_name, last_name, hand, dob, ioc FROM atp_players"
    )
    players = cur.fetchall()
    by_exact = defaultdict(list)
    by_norm = defaultdict(list)
    for pid, fn, ln, hand, dob, ioc in players:
        full = f"{fn} {ln}"
        by_exact[full].append((pid, hand, dob, ioc))
        by_norm[norm_name(full)].append((pid, hand, dob, ioc))
    cur.execute(
        "SELECT DISTINCT winner_id FROM atp_matches WHERE tourney_date >= '2025-01-01'"
    )
    used_ids = {r[0] for r in cur.fetchall()}
    cur.execute(
        "SELECT DISTINCT loser_id FROM atp_matches WHERE tourney_date >= '2025-01-01'"
    )
    used_ids |= {r[0] for r in cur.fetchall()}
    max_pid = max(p[0] for p in players)

    def disambiguate(cands, ioc, age, match_date):
        if len(cands) == 1:
            return cands[0][0]
        used = [c for c in cands if c[0] in used_ids]
        if len(used) == 1:
            return used[0][0]
        pool = used if used else cands
        if ioc:
            by_ioc = [c for c in pool if c[3] == ioc]
            if len(by_ioc) == 1:
                return by_ioc[0][0]
            if by_ioc:
                pool = by_ioc
        if age and match_date:
            best = None
            for c in pool:
                if c[2]:
                    dob = date.fromisoformat(c[2])
                    exp_age = (match_date - dob).days / 365.25
                    err = abs(exp_age - float(age))
                    if best is None or err < best[0]:
                        best = (err, c)
            if best and best[0] < 1.0:
                return best[1][0]
        with_hand = [c for c in pool if c[1] != "U"]
        if len(with_hand) == 1:
            return with_hand[0][0]
        if with_hand:
            pool = with_hand
        return sorted(pool, key=lambda c: c[0])[0][0]

    def resolve(name, ioc, age, match_date):
        db_name = NAME_OVERRIDES.get(name, name)
        cands = by_exact.get(db_name)
        if cands:
            return disambiguate(cands, ioc, age, match_date), "exact"
        cands = by_norm.get(norm_name(db_name))
        if cands:
            return disambiguate(cands, ioc, age, match_date), "norm"
        nn = norm_name(db_name)
        ntok = nn.split()
        # prefix (either direction) or consecutive-token subsequence
        pref = []
        for key, c in by_norm.items():
            ktok = key.split()
            if key.startswith(nn + " ") or nn.startswith(key + " "):
                pref.extend(c)
                continue
            for i in range(len(ktok) - len(ntok) + 1):
                if ktok[i:i + len(ntok)] == ntok:
                    pref.extend(c)
                    break
            else:
                for i in range(len(ntok) - len(ktok) + 1):
                    if ntok[i:i + len(ktok)] == ktok:
                        pref.extend(c)
                        break
        if pref:
            seen = {}
            for c in pref:
                seen[c[0]] = c
            return disambiguate(list(seen.values()), ioc, age, match_date), "subseq"
        toks = db_name.split()
        if len(toks) >= 2:
            rev = " ".join([toks[-1]] + toks[:-1])
            cands = by_exact.get(rev) or by_norm.get(norm_name(rev))
            if cands:
                return disambiguate(cands, ioc, age, match_date), "reversed"
        return None, "unresolved"

    new_players = {}
    resolve_report = Counter()
    unresolved_names = Counter()
    resolved_ids = {}
    for r in rows:
        md = date.fromisoformat(r["tourney_date"])
        for who, ioc_k, age_k, hand_k, ht_k in (
            ("winner_name", "winner_ioc", "winner_age", "winner_hand", "winner_ht"),
            ("loser_name", "loser_ioc", "loser_age", "loser_hand", "loser_ht"),
        ):
            name = r[who]
            if name in resolved_ids or name in new_players:
                continue
            pid, method = resolve(name, r[ioc_k], r[age_k], md)
            if pid is None:
                unresolved_names[name] += 1
                toks = name.split()
                new_players[name] = (
                    " ".join(toks[:-1]), toks[-1], r[hand_k], r[ioc_k], r[ht_k],
                )
            else:
                resolved_ids[name] = pid
                resolve_report[method] += 1
    print(f"names resolved: {len(resolved_ids)} by {dict(resolve_report)}")
    print(f"unresolved (new players): {len(new_players)}")
    for n, c in sorted(unresolved_names.items()):
        print(f"  NEW {n} x{c}")

    def parse_seed(seed_val, entry_val):
        """TML sometimes puts 'Q' in seed; DB wants it in entry."""
        if not seed_val:
            return None, entry_val if entry_val else None
        if seed_val.isdigit():
            return int(seed_val), entry_val if entry_val else None
        return None, entry_val if entry_val else seed_val

    to_insert = []
    dup = 0
    for r in rows:
        tid_stripped = strip_tid(r["tourney_id"])
        level = LEVEL_MAP[r["tourney_level"]]
        wid = resolved_ids.get(r["winner_name"])
        lid = resolved_ids.get(r["loser_name"])
        if wid is not None and lid is not None:
            if level == "D":
                key = (norm_tie_id(r["tourney_id"]), wid, lid)
                if key in db_d_keys:
                    dup += 1
                    continue
            else:
                key = (tid_stripped, wid, lid)
                if key in db_keys:
                    dup += 1
                    continue
        # rows involving a brand-new player cannot duplicate a DB row
        # (the player has no DB record at all)
        vals = {}
        vals["tourney_id"] = db_tid_by_stripped.get(
            tid_stripped,
            TOURNEY_ID_MAP.get(r["tourney_id"], r["tourney_id"]),
        )
        vals["tourney_name"] = TOURNEY_NAME_MAP.get(
            r["tourney_name"], r["tourney_name"]
        )
        vals["surface"] = r["surface"]
        vals["draw_size"] = int(r["draw_size"]) if r["draw_size"] else None
        vals["tourney_level"] = level
        vals["tourney_date"] = monday[r["tourney_id"]].isoformat()
        vals["match_num"] = int(r["match_num"])
        for who in ("winner", "loser"):
            name = r[f"{who}_name"]
            vals[f"{who}_id"] = resolved_ids.get(name)  # None -> new player
            seed, entry = parse_seed(r.get(f"{who}_seed", ""),
                                     r.get(f"{who}_entry", ""))
            vals[f"{who}_seed"] = seed
            vals[f"{who}_entry"] = entry
            vals[f"{who}_name"] = name
            vals[f"{who}_hand"] = r[f"{who}_hand"] or None
            ht = r.get(f"{who}_ht", "")
            vals[f"{who}_ht"] = int(ht) if ht else None
            vals[f"{who}_ioc"] = r[f"{who}_ioc"] or None
            age = r.get(f"{who}_age", "")
            vals[f"{who}_age"] = float(age) if age else None
            rk = r.get(f"{who}_rank", "")
            vals[f"{who}_rank"] = int(rk) if rk else None
            rp = r.get(f"{who}_rank_points", "")
            vals[f"{who}_rank_points"] = int(rp) if rp else None
        vals["score"] = r["score"]
        vals["best_of"] = int(r["best_of"]) if r["best_of"] else None
        vals["round"] = r["round"]
        vals["minutes"] = int(r["minutes"]) if r["minutes"] else None
        for tml_col, db_col in COL_MAP.items():
            src = r.get(tml_col, "")
            vals[db_col] = int(src) if src else None
        for col in ("w_ace", "w_df", "w_svpt", "l_ace", "l_df", "l_svpt"):
            src = r.get(col, "")
            vals[col] = int(src) if src else None
        to_insert.append((r, vals))
    print(f"rows to insert: {len(to_insert)} | duplicates skipped: {dup}")
    print("  by level:", dict(Counter(v["tourney_level"] for _, v in to_insert)))
    print("  by month:", dict(sorted(Counter(
        v["tourney_date"][:7] for _, v in to_insert).items())))
    keys = Counter((v["tourney_id"], v["match_num"]) for _, v in to_insert)
    collisions = {k: c for k, c in keys.items() if c > 1}
    print(f"  internal (tid, match_num) collisions: {len(collisions)}")
    for k, c in list(collisions.items())[:10]:
        print("   ", k, c)
    # atp_matches has a UNIQUE (tourney_id, match_num) constraint, and the
    # DB numbers some tournaments differently from TML (e.g. the PAR-ROU
    # Davis Cup tie: DB 1,2,4 vs TML 1,2,3). For any tournament that
    # already has DB rows, renumber inserted rows from max+1, preserving
    # TML order, so they neither collide nor reorder.
    wrong_set = {(t, w, l) for t, w, l in WRONG_ROWS}
    cur.execute(
        "SELECT tourney_id, match_num, winner_id, loser_id FROM atp_matches"
    )
    existing_mn = defaultdict(list)
    for tid, mn, wid, lid in cur.fetchall():
        if (tid, wid, lid) in wrong_set:
            continue  # these rows are deleted on write
        existing_mn[tid].append(mn)
    renumbered = 0
    by_tid = defaultdict(list)
    for r, v in to_insert:
        by_tid[v["tourney_id"]].append((r, v))
    for tid, items in by_tid.items():
        if tid in existing_mn:
            nxt = max(existing_mn[tid]) + 1
            for r, v in sorted(items, key=lambda x: x[1]["match_num"]):
                old = v["match_num"]
                v["match_num"] = nxt
                nxt += 1
                renumbered += 1
                print(f"  renumber {tid}: {old} -> {v['match_num']} "
                      f"({v['winner_name']} d. {v['loser_name']})")
    print(f"  renumbered rows (existing tournament): {renumbered}")
    # any inserted row whose tournament already exists in DB? (should be none except RG-like)
    existing_tids = {strip_tid(t) for t in db_tid_by_stripped}
    overlap = Counter(v["tourney_name"] for _, v in to_insert
                      if strip_tid(v["tourney_id"]) in existing_tids)
    print("  inserted rows in tournaments already in DB:", dict(overlap))

    if args.dry_run:
        print("DRY RUN: nothing written.")
        return

    shutil.copyfile(args.db, args.db + ".bak-20261124")
    print(f"backup written: {args.db}.bak-20261124")

    # correct the two verified wrong-ID rows (see WRONG_ROWS) so the
    # correct TML versions can insert without duplicating the tie.
    for tid, wid, lid in WRONG_ROWS:
        cur.execute(
            "DELETE FROM atp_matches WHERE tourney_id=? AND winner_id=? AND loser_id=?",
            (tid, wid, lid),
        )
        print(f"corrected: deleted wrong row {tid} {wid} d. {lid} "
              f"(rows removed: {cur.rowcount})")

    new_id_map = {}
    for name, (fn, ln, hand, ioc, ht) in sorted(new_players.items()):
        if not any(r["winner_name"] == name or r["loser_name"] == name
                   for r, _ in to_insert):
            continue
        max_pid += 1
        cur.execute(
            "INSERT INTO atp_players (player_id, first_name, last_name, hand, "
            "dob, ioc, height_cm, wikidata_id) VALUES (?, ?, ?, ?, NULL, ?, ?, NULL)",
            (max_pid, fn, ln, hand if hand else "U", ioc if ioc else None,
             int(ht) if ht else None),
        )
        new_id_map[name] = max_pid
    print(f"new players inserted: {len(new_id_map)}")

    inserted = 0
    for r, vals in to_insert:
        for who in ("winner", "loser"):
            if vals[f"{who}_id"] is None:
                vals[f"{who}_id"] = new_id_map[r[f"{who}_name"]]
        cols = ", ".join(MATCH_COLS)
        ph = ", ".join(["?"] * len(MATCH_COLS))
        cur.execute(
            f"INSERT INTO atp_matches ({cols}) VALUES ({ph})",
            [vals[c] for c in MATCH_COLS],
        )
        inserted += 1
    con.commit()
    cur.execute(
        "SELECT COUNT(*), MIN(tourney_date), MAX(tourney_date) FROM atp_matches"
    )
    print(f"inserted {inserted} rows. DB now: {cur.fetchone()}")
    cur.execute(
        "SELECT COUNT(*) FROM atp_matches WHERE tourney_date >= '2026-06-01'"
    )
    print(f"2026 H2 rows in DB: {cur.fetchone()[0]}")
    con.close()


if __name__ == "__main__":
    main()
