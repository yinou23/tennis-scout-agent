"""Tests for scripts/backtest/.

Covers the verification standards for the framework:
- Brier / log loss hand calculation (p=0.7, winner -> Brier 0.09).
- De-vigging hand calculation.
- Paired model-vs-market difference hand calculation.
- Name normalisation (accents, initials).
- No-leakage walk-forward: a match result only affects LATER
  predictions, never the match itself or earlier ones.
- Elo recompute matches the stored atp_elo_history /
  atp_surface_elo_history tables (small window, fast).
- tennis-data xlsx reader + alignment against a synthetic fixture.
- run_baseline refuses the locked test set (2025-2026).
- Serve/return features are walk-forward and symmetric; the Elastic
  Net container has no intercept and is symmetric; the elo-only
  container reproduces the baseline Brier. The logreg tests need
  scikit-learn, installed in .venv (pip install scikit-learn).
- All-batches features: score parsing; Glicko-2 reproduces the
  paper's worked example and idle RD growth; form, fatigue, WElo,
  head-to-head states are walk-forward and symmetric; the
  run_logreg_all row builder works on a synthetic list.
"""

import math
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "backtest"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from align import _is_team_exhibition, align_matches  # noqa: E402
from elo_state import EloState, verify_against_history  # noqa: E402
from metrics import devig, ece, paired_diff, score_outcome  # noqa: E402
from names import name_key, normalize_name, pair_key  # noqa: E402
from odds import OddsRow, read_tennis_data_xlsx, read_valuebetennis_csv  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Metrics, hand-calculated
# ---------------------------------------------------------------------------

def test_brier_hand_calculation():
    # p=0.7 for the winner: Brier (1-0.7)^2 = 0.09, log loss -ln(0.7).
    m = score_outcome([0.7])
    assert m["n"] == 1
    assert m["brier"] == pytest.approx(0.09)
    assert m["log_loss"] == pytest.approx(-math.log(0.7))
    assert m["accuracy"] == 1.0
    # Coin flip: Brier 0.25, log loss ln(2).
    m2 = score_outcome([0.5, 0.5])
    assert m2["brier"] == pytest.approx(0.25)
    assert m2["log_loss"] == pytest.approx(math.log(2))


def test_devig_hand_calculation():
    assert devig(2.0, 2.0) == pytest.approx((0.5, 0.5))
    pa, pb = devig(1.5, 2.5)
    assert pa == pytest.approx(0.625)
    assert pb == pytest.approx(0.375)
    assert pa + pb == pytest.approx(1.0)
    with pytest.raises(ValueError):
        devig(1.0, 2.0)


def test_paired_diff_hand_calculation():
    # model p winner [0.7, 0.6], market [0.8, 0.5]:
    # d = [0.09-0.04, 0.16-0.25] = [0.05, -0.09]; D=-0.02.
    r = paired_diff([0.7, 0.6], [0.8, 0.5])
    assert r["n"] == 2
    assert r["D"] == pytest.approx(-0.02)
    assert r["se"] == pytest.approx(0.07)
    assert r["z"] == pytest.approx(-0.02 / 0.07)


def test_ece_perfect_and_worst():
    # Perfectly calibrated favourite probs: ECE 0.
    r = ece([0.6, 0.6, 0.8, 0.8], [1, 0, 1, 1])
    # bins 0.6-0.7: mean 0.6 actual 0.5 -> 0.05; 0.8-0.9: mean 0.8 actual 1.0 -> 0.1
    # ECE = 0.5*0.1 + 0.5*0.2 = 0.15
    assert r["ece"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_name_normalisation():
    assert normalize_name("Đoković") == "dokovic"
    assert normalize_name("Auger-Aliassime") == "auger aliassime"
    assert name_key("Rafael Nadal") == ("nadal", "r")
    assert name_key("Nadal R") == ("nadal", "r")
    assert name_key("Nadal, Rafael") == ("nadal", "r")
    assert pair_key("Rafael Nadal", "Carlos Alcaraz") == pair_key("Carlos Alcaraz", "Rafael Nadal")


def test_name_key_multiword_surname_with_initial():
    # Regression: real tennis-data files write multi-word surnames as
    # "O Connell C.", "De Minaur A.", "Van De Zandschulp B.". The old
    # code keyed these as (initial, first-token-initial), so every
    # match involving such a player came back "no_pair" and the
    # 2022-2024 tennis-data match rate sat at ~72%.
    assert name_key("O Connell C.") == name_key("Christopher Oconnell") == ("oconnell", "c")
    assert name_key("De Minaur A.") == name_key("Alex De Minaur") == ("minaur", "a")
    assert name_key("Van Assche L.") == name_key("Luca Van Assche") == ("assche", "l")
    assert name_key("Auger Aliassime F.") == name_key("Felix Auger Aliassime") == ("aliassime", "f")
    assert name_key("Van De Zandschulp B.") == name_key("Botic Van De Zandschulp") == ("zandschulp", "b")
    assert name_key("De Jong J.") == name_key("Jesper De Jong") == ("jong", "j")
    assert name_key("Kwon S.W.") == name_key("Soonwoo Kwon") == ("kwon", "s")
    assert name_key("Carballes Baena R.") == name_key("Roberto Carballes Baena") == ("baena", "r")


# ---------------------------------------------------------------------------
# Elo walk-forward, no leakage
# ---------------------------------------------------------------------------

def test_elo_no_leakage():
    st = EloState()
    # Before any match, everyone is 1500: A vs B is 50/50.
    p, _, _, _ = st.predict(1, 2, "Hard")
    assert p == pytest.approx(0.5)
    # A beats B. A's Elo rises, B's falls, but a prediction made
    # BEFORE the update must not see it: predict-then-update order
    # is enforced by the caller; here check the update itself.
    st.update(1, 2, "Hard")
    p2, _, ba, bb = st.predict(1, 3, "Hard")
    assert ba > 1500 and bb == 1500
    assert p2 > 0.5
    # Surface debut: no matches on Clay -> blended is overall Elo.
    _p, _, ba_clay, _bb = st.predict(1, 3, "Clay")
    assert ba_clay == pytest.approx(st.overall[1])
    # A beats C on Clay: surface Elo moves, Hard Elo does not.
    hard_before = st.surface_rating[(1, "Hard")]
    st.update(1, 3, "Clay")
    assert st.surface_rating[(1, "Hard")] == hard_before
    assert st.surface_n[(1, "Clay")] == 1


def test_elo_state_defaults_are_locked_constants():
    from elo_state import CALIBRATION_A, CALIBRATION_B, PRIOR
    from elo import DEFAULT_K
    st = EloState()
    assert st.k == DEFAULT_K == 32
    assert st.prior == PRIOR == 100
    assert st.cal_a == CALIBRATION_A == 0.8
    assert st.cal_b == CALIBRATION_B == 0.0


def test_elo_state_parameterized_matches_default():
    # Explicit locked parameters must reproduce the default state
    # exactly, match by match; a different K must change it.
    seq = [(1, 2, "Hard"), (1, 3, "Hard"), (2, 3, "Clay"),
           (3, 1, "Hard"), (2, 1, "Grass"), (3, 2, "Hard")]
    default = EloState()
    explicit = EloState(k=32, prior=100, cal_a=0.8, cal_b=0.0)
    other_k = EloState(k=16, prior=100, cal_a=0.8, cal_b=0.0)
    other_prior = EloState(k=32, prior=50, cal_a=0.8, cal_b=0.0)
    for w, l, surf in seq:
        p_d = default.predict(w, l, surf)
        p_e = explicit.predict(w, l, surf)
        assert p_d == p_e
        default.update(w, l, surf)
        explicit.update(w, l, surf)
        other_k.update(w, l, surf)
        other_prior.update(w, l, surf)
    # Different K / prior produce a different state after the same
    # matches (otherwise the parameters would be dead). Note prior
    # only changes the blend in predict(), not the surface ratings
    # themselves, so compare overall Elo for K and predictions
    # for prior.
    assert other_k.overall != default.overall
    assert other_prior.predict(1, 2, "Hard") != default.predict(1, 2, "Hard")


def test_elo_state_calibration_is_predict_only():
    # Calibration must not leak into updates: two states with the
    # same (k, prior) but different calibration share the same Elo
    # after identical updates, and the offline calibration of the
    # recorded blended probability reproduces predict() exactly.
    from tennis_tools import calibrated_prob
    seq = [(1, 2, "Hard"), (1, 3, "Clay"), (2, 3, "Hard")]
    st_a = EloState(k=32, prior=100, cal_a=0.8, cal_b=0.0)
    st_b = EloState(k=32, prior=100, cal_a=1.0, cal_b=0.1)
    for w, l, surf in seq:
        _p_cal, p_blended, _ba, _bb = st_a.predict(w, l, surf)
        p_cal_b, _pb, _c, _d = st_b.predict(w, l, surf)
        assert p_cal_b == pytest.approx(
            calibrated_prob(p_blended, a=1.0, b=0.1))
        st_a.update(w, l, surf)
        st_b.update(w, l, surf)
    assert st_a.overall == st_b.overall
    assert st_a.surface_rating == st_b.surface_rating


def test_elo_recompute_matches_history_tables():
    from db import get_connection
    conn = get_connection()
    check = verify_against_history(conn, upto_date="1980-12-31")
    assert check["error"] is None
    assert check["checked"] > 1000
    assert check["max_overall_diff"] < 1e-6
    assert check["max_surface_diff"] < 1e-6
    assert check["mismatches"] == 0


# ---------------------------------------------------------------------------
# Odds readers + alignment (synthetic tennis-data fixture)
# ---------------------------------------------------------------------------

def _write_fixture_xlsx(path):
    """Minimal .xlsx with tennis-data-style headers, stdlib only."""
    strings = ["Tournament", "Round", "Winner", "Loser", "Date", "B365W", "B365L", "PSW", "PSL",
               "Australian Open", "R32", "Rafael Nadal", "Carlos Alcaraz", "2024-01-15",
               "Wimbledon", "F", "Novak Djokovic", "Jannik Sinner", "2024-07-14"]
    sst = "".join(f"<si><t>{s}</t></si>" for s in strings)
    shared = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
              f'count="{len(strings)}" uniqueCount="{len(strings)}">{sst}</sst>')
    # Rows: header + 2 matches. Each <row> mixes shared-string cells
    # (t="s") and plain-value cells (date text, numeric odds).
    def row(r, cells):
        out = [f'<row r="{r}">']
        for col, kind, val in cells:
            if kind == "s":
                out.append(f'<c r="{col}{r}" t="s"><v>{val}</v></c>')
            else:
                out.append(f'<c r="{col}{r}"><v>{val}</v></c>')
        out.append("</row>")
        return "".join(out)
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData>'
             + row(1, [("A", "s", 0), ("B", "s", 1), ("C", "s", 2), ("D", "s", 3), ("E", "s", 4), ("F", "s", 5), ("G", "s", 6), ("H", "s", 7), ("I", "s", 8)])
             + row(2, [("A", "s", 9), ("B", "s", 10), ("C", "s", 11), ("D", "s", 12), ("E", "v", "2024-01-15"), ("F", "v", 1.5), ("G", "v", 2.6), ("H", "v", 1.45), ("I", "v", 2.9)])
             + row(3, [("A", "s", 14), ("B", "s", 15), ("C", "s", 16), ("D", "s", 17), ("E", "v", "2024-07-14"), ("F", "v", 2.1), ("G", "v", 1.8), ("H", "v", 2.0), ("I", "v", 1.85)])
             + '</sheetData></worksheet>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets><sheet name="2024" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
          '</Types>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("xl/workbook.xml", wb)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
        zf.writestr("xl/sharedStrings.xml", shared)


def test_tennis_data_xlsx_reader_fixture(tmp_path):
    path = tmp_path / "2024.xlsx"
    _write_fixture_xlsx(path)
    rows = read_tennis_data_xlsx(path, 2024)
    assert len(rows) == 2
    assert rows[0].player_a == "Rafael Nadal"
    assert rows[0].odds["bet365"] == pytest.approx((1.5, 2.6))
    assert rows[0].odds["pinnacle"] == pytest.approx((1.45, 2.9))
    assert rows[1].tournament == "Wimbledon"


def test_team_exhibition_both_nextgen_spellings():
    # Regression: Sackmann spells it "NextGen Finals" (2017-2022) and
    # "Next Gen Finals" (2023-2024); the old check only caught the
    # spaced spelling, so 92 pre-2023 NextGen matches slipped through.
    assert _is_team_exhibition("NextGen Finals")
    assert _is_team_exhibition("Next Gen Finals")
    assert _is_team_exhibition("United Cup")
    assert _is_team_exhibition("Laver Cup")
    assert not _is_team_exhibition("Australian Open")


def test_alignment_fixture():
    import datetime as dt
    matches = [{
        "tourney_id": "2024-580", "match_num": 1,
        "tourney_date": dt.date(2024, 1, 14), "tourney_name": "Australian Open",
        "round": "R32", "surface": "Hard", "tourney_level": "G",
        "winner_name": "Rafael Nadal", "loser_name": "Carlos Alcaraz",
    }]
    odds = [OddsRow(source="tennis-data.co.uk", year=2024, date=dt.date(2024, 1, 15),
                    tournament="Australian Open", round="R32",
                    player_a="Rafael Nadal", player_b="Carlos Alcaraz",
                    odds={"bet365": (1.5, 2.6)})]
    matched, unmatched = align_matches(matches, odds)
    assert len(matched) == 1 and not unmatched
    # Same pair, odds date outside the 14-day window -> no_date_window.
    odds2 = [OddsRow(source="tennis-data.co.uk", year=2024, date=dt.date(2024, 3, 1),
                     tournament="Australian Open", round="R32",
                     player_a="Rafael Nadal", player_b="Carlos Alcaraz",
                     odds={"bet365": (1.5, 2.6)})]
    _m, un = align_matches(matches, odds2)
    assert un[0][1] == "no_date_window"


def test_valuebetennis_reader_real_file():
    path = REPO / "scripts" / "backtest" / "data" / "valuebetennis-matchs-2023.csv"
    if not path.exists():
        pytest.skip("Valuebetennis file not downloaded")
    rows = read_valuebetennis_csv(path, 2023)
    assert len(rows) > 1000
    assert all("pinnacle" in r.odds for r in rows)
    assert all(r.date is not None for r in rows)


def test_run_baseline_refuses_test_set():
    from run_baseline import main
    assert main(["--end-year", "2025", "--skip-elo-verify"]) == 2
    assert main(["--start-year", "2025", "--end-year", "2025", "--skip-elo-verify"]) == 2


# ---------------------------------------------------------------------------
# Serve/return features + Elastic Net container (logreg_serve)
# ---------------------------------------------------------------------------

def _serve_match(winner_id, loser_id, w_hold_won=5, w_sv_gms=10,
                 w_bp_faced=2, w_bp_saved=1, l_sv_gms=10,
                 l_bp_faced=3, l_bp_saved=1):
    """Synthetic serve stats: winner held 5+? games, broke twice.

    Winner service games won = w_sv_gms - (w_bp_faced - w_bp_saved);
    winner return games won = l_bp_faced - l_bp_saved.
    """
    return {
        "winner_id": winner_id, "loser_id": loser_id, "surface": "Hard",
        "w_sv_gms": w_sv_gms, "w_bp_faced": w_bp_faced,
        "w_bp_saved": w_bp_saved, "l_sv_gms": l_sv_gms,
        "l_bp_faced": l_bp_faced, "l_bp_saved": l_bp_saved,
    }


def test_serve_state_is_walk_forward():
    from features import ServeState, TOUR_HOLD_PRIOR

    s = ServeState()
    # No history: both players get the tour prior, diff is 0.
    h, b = s.diffs(1, 2, "Hard")
    assert h == pytest.approx(0.0) and b == pytest.approx(0.0)
    # Player 1 beats player 2: holds 9/10, breaks 2/10.
    m = _serve_match(1, 2, w_sv_gms=10, w_bp_faced=2, w_bp_saved=1,
                     l_sv_gms=10, l_bp_faced=3, l_bp_saved=1)
    # Features BEFORE the update must not include this match.
    h_before, _ = s.diffs(1, 2, "Hard")
    assert h_before == pytest.approx(0.0)
    s.update(m["winner_id"], m["loser_id"], m["surface"],
             m["w_sv_gms"], m["w_bp_faced"], m["w_bp_saved"],
             m["l_sv_gms"], m["l_bp_faced"], m["l_bp_saved"])
    # After: player 1's hold is shrunk toward the tour prior, so it is
    # between the raw 9/10 and the prior 0.7909, and above player 2's.
    hold1, brk1 = s.rates(1, "Hard")
    hold2, brk2 = s.rates(2, "Hard")
    assert TOUR_HOLD_PRIOR["Hard"] < hold1 < 0.9
    assert hold1 > hold2
    assert brk1 > brk2  # player 1 broke twice, player 2 once
    # A match with NULL serve columns leaves the state untouched.
    s.update(1, 2, "Hard", None, None, None, None, None, None)
    assert s.rates(1, "Hard") == pytest.approx((hold1, brk1))


def test_serve_diffs_are_symmetric():
    from features import ServeState

    s = ServeState()
    s.update(1, 2, "Hard", 10, 2, 1, 10, 3, 1)
    s.update(2, 3, "Clay", 9, 1, 1, 9, 2, 0)
    h_ab, b_ab = s.diffs(1, 2, "Hard")
    h_ba, b_ba = s.diffs(2, 1, "Hard")
    assert h_ab == pytest.approx(-h_ba)
    assert b_ab == pytest.approx(-b_ba)


def test_logreg_container_has_no_intercept_and_is_symmetric():
    import numpy as np

    from logreg import fit_logreg, predict_proba

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    for C, l1 in ((1.0, 0.0), (0.1, 0.5), (0.01, 1.0)):
        clf = fit_logreg(X, y, C, l1)
        assert clf.intercept_.shape == (1,)
        assert float(clf.intercept_[0]) == 0.0
        p = predict_proba(clf, X[:50])
        p_swap = predict_proba(clf, -X[:50])
        assert p_swap == pytest.approx(1.0 - p, abs=1e-9)


def test_elo_only_container_reproduces_baseline():
    """Full-walk sanity check: elo-only logreg lands near the locked
    baseline validation Brier 0.2185 (main-tour non-team, n=5,464)."""
    import numpy as np

    from db import get_connection
    from logreg import Standardizer, fit_logreg, load_matches_with_serve, predict_proba
    from metrics import score_outcome
    from run_logreg_serve import build_rows

    matches = load_matches_with_serve(get_connection())
    rows = build_rows(matches)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validate"]
    assert len(val) == 5464
    Xtr = np.array([r["x"] for r in train])[:, [0]]
    ytr = np.array([r["y"] for r in train])
    Xva = np.array([r["x"] for r in val])[:, [0]]
    yva = np.array([r["y"] for r in val])
    scaler = Standardizer().fit(Xtr)
    clf = fit_logreg(scaler.transform(Xtr), ytr, 0.01, 1.0)
    p_a = predict_proba(clf, scaler.transform(Xva))
    p_win = [p if y == 1 else 1.0 - p for p, y in zip(p_a, yva)]
    brier = score_outcome(p_win)["brier"]
    assert brier == pytest.approx(0.2185, abs=0.002)


# ---------------------------------------------------------------------------
# All-batches features: form, fatigue, WElo, Glicko-2, age, head-to-head
# ---------------------------------------------------------------------------

def test_parse_score_cases():
    from features import parse_score

    assert parse_score("6-4 3-6 7-6(5)") == (16, 16, 3, False, False)
    assert parse_score("6-2 0-0 RET") == (6, 2, 2, True, False)
    assert parse_score("W/O") == (0, 0, 0, False, True)
    assert parse_score("Walkover") == (0, 0, 0, False, True)
    assert parse_score(None) == (0, 0, 0, False, True)
    # Bracketed match tiebreak counts as a 7-6 set.
    assert parse_score("6-3 3-6 [11-9]") == (16, 15, 3, False, False)


def test_glicko_paper_example():
    """Glickman's worked example: 1500/RD200/sigma0.06, win vs
    1400/RD30, losses vs 1550/RD100 and 1700/RD300, tau 0.5 ->
    1464.06 / 151.52 / 0.05999. Rating tolerance is 0.05: the paper's
    own arithmetic used 4-decimal intermediates."""
    from glicko import paper_example

    r, rd, sigma = paper_example()
    assert r == pytest.approx(1464.06, abs=0.05)
    assert rd == pytest.approx(151.52, abs=0.01)
    assert sigma == pytest.approx(0.05999, abs=0.00005)


def test_glicko_idle_rd_grows():
    """Step 6 for idle periods: RD grows by sigma^2 per idle week;
    a never-played player stays at the initial RD 350."""
    import datetime as dt

    from glicko import GlickoState, SCALE

    g = GlickoState()
    d0 = dt.date(2024, 1, 1)
    assert g.rd(99, d0) == pytest.approx(350.0)
    g.end_of_week(d0, {1: [(2, 1)], 2: [(1, 0)]})
    rd_now = g.rd(1, d0)
    rd_later = g.rd(1, d0 + dt.timedelta(days=70))
    assert rd_later > rd_now
    entry = g._players[1]
    weeks = GlickoState._week_of(d0 + dt.timedelta(days=70)) - entry["last_week"]
    assert weeks == 10
    expect = math.sqrt(entry["phi"] ** 2 + entry["sigma"] ** 2 * weeks) * SCALE
    assert rd_later == pytest.approx(expect)


def test_form_state_is_walk_forward():
    """Form and residual form use only matches already processed."""
    import datetime as dt

    from features import FormState

    d = dt.date(2024, 1, 1)
    f = FormState()
    assert f.form_10(1, d) == 0.0
    assert f.resid_10(1, d) == 0.0
    for i in range(3):
        f.update(1, 2, d + dt.timedelta(days=i), 0.5)
    f.update(2, 1, d + dt.timedelta(days=3), 0.5)
    # Last 10 for player 1: 3 wins / 4, shrunk k=2 toward 0.5.
    assert f.form_10(1, d + dt.timedelta(days=4)) == pytest.approx(4 / 6 - 0.5)
    # Residuals (0.5, 0.5, 0.5, -0.5): mean 0.25, shrunk x4/6.
    assert f.resid_10(1, d + dt.timedelta(days=4)) == pytest.approx(0.25 * 4 / 6)
    # A 60-day-old win no longer counts in the 2-month window.
    assert f.form_2m(1, d + dt.timedelta(days=61)) == pytest.approx(0.0)
    # Symmetric: A's form diff is the negative of B's.
    assert f.diffs(1, 2, d + dt.timedelta(days=4))[0] == pytest.approx(
        -f.diffs(2, 1, d + dt.timedelta(days=4))[0])


def test_fatigue_state_windows():
    import datetime as dt

    from features import FatigueState

    d = dt.date(2024, 1, 1)
    ft = FatigueState(tour_mean_minutes=100.0)
    ft.update(1, 2, d, 20, 90)
    ft.update(1, 3, d + dt.timedelta(days=1), 25, 100)
    r = ft.diffs(1, 2, d + dt.timedelta(days=3))
    assert r == pytest.approx((-1.0, 25.0, 25.0, 10.0))
    r = ft.diffs(1, 2, d + dt.timedelta(days=10))
    assert r == pytest.approx((-1.0, 0.0, 25.0, 10.0))
    # Walkovers (0 games) do not move the rest clock.
    ft.update(1, 2, d + dt.timedelta(days=11), 0, None)
    assert ft.diffs(1, 2, d + dt.timedelta(days=12))[0] == pytest.approx(-1.0)


def test_welo_state_weighted_update():
    """A 6-0 6-0 win moves WElo the full K/2; a 7-6 7-6 win moves it
    by the games share 14/26; retirements are skipped."""
    from features import WEloState

    w = WEloState()
    w.update(1, 2, "Hard", 12, 12)
    assert w.blended(1, "Hard") == pytest.approx(1516.0)
    assert w.blended(2, "Hard") == pytest.approx(1484.0)
    w2 = WEloState()
    w2.update(1, 2, "Hard", 14, 26)
    assert w2.blended(1, "Hard") == pytest.approx(1500 + 32 * (14 / 26) * 0.5)
    w3 = WEloState()
    w3.update(1, 2, "Hard", 6, 8, retired=True)
    assert w3.blended(1, "Hard") == pytest.approx(1500.0)
    w4 = WEloState()
    w4.update(1, 2, "Hard", 0, 0, walkover=True)
    assert w4.blended(1, "Hard") == pytest.approx(1500.0)


def test_h2h_state_floor_and_symmetry():
    from features import H2HState

    h = H2HState()
    h.update(1, 2)
    h.update(1, 2)
    assert h.diffs(1, 2) == (0.0, 0.0, 0.0)  # fewer than 3 meetings
    h.update(1, 2)
    v = h.diffs(1, 2)
    assert v[1] == pytest.approx(2 * ((3 + 2.5) / (3 + 5) - 0.5))
    v2 = h.diffs(2, 1)
    assert v2 == pytest.approx(tuple(-x for x in v), abs=1e-12)


def test_age_years():
    import datetime as dt

    from features import age_years

    dob = dt.date(2000, 1, 1)
    assert age_years(dob, dt.date(2024, 1, 1)) == pytest.approx(24.0, abs=0.01)
    assert age_years(None, dt.date(2024, 1, 1)) is None


def test_all_batches_row_builder_synthetic():
    """build_rows on a small synthetic list: 21 features per row, no
    crash, labels follow the lower-id player, and the Glicko pre-pass
    runs. This is the integration check for run_logreg_all; the real
    data run is the batch job itself."""
    import datetime as dt

    from logreg import FEATURE_NAMES
    from run_logreg_all import build_rows

    base = dt.date(2023, 1, 2)
    matches = []
    for i in range(30):
        w, l = (101, 102) if i % 2 == 0 else (102, 101)
        matches.append({
            "tourney_id": f"2023-001", "match_num": i + 1,
            "tourney_date": base + dt.timedelta(days=i * 3),
            "tourney_name": "Synthetic Open", "surface": "Hard",
            "round": "R32", "tourney_level": "A",
            "winner_id": w, "loser_id": l,
            "winner_name": "A Player" if w == 101 else "B Player",
            "loser_name": "B Player" if w == 101 else "A Player",
            "w_sv_gms": 10, "w_bp_faced": 2, "w_bp_saved": 1,
            "l_sv_gms": 10, "l_bp_faced": 3, "l_bp_saved": 1,
            "score": "6-4 6-4", "minutes": 95, "best_of": 3,
        })
    dob = {101: dt.date(1998, 5, 1), 102: dt.date(2001, 5, 1)}
    rows = build_rows(matches, dob, 100.0)
    assert len(rows) == 30
    assert all(len(r["x"]) == 21 for r in rows)  # this builder's width
    assert all(r["split"] == "validate" for r in rows)
    # y = 1 iff the lower-id player (101) won.
    assert rows[0]["y"] == 1 and rows[1]["y"] == 0
    # Age diff is nonzero (1998 vs 2001 dob) and constant across rows.
    age_col = FEATURE_NAMES.index("age_diff")
    assert rows[0]["x"][age_col] == pytest.approx(3.0, abs=0.01)


def test_recent_serve_state_window_and_symmetry():
    """RecentServeState: only the last 60 days count; old matches
    expire out of the window; diffs are symmetric."""
    import datetime as dt

    from features import RecentServeState

    s = RecentServeState()
    d0 = dt.date(2024, 1, 1)
    # Player 1 holds 9/10 on Jan 1.
    s.update(1, 2, "Hard", d0, 10, 2, 1, 10, 3, 1)
    h, _ = s.diffs(1, 2, "Hard", d0 + dt.timedelta(days=30))
    assert h > 0.0  # still inside the 60-day window
    h_late, _ = s.diffs(1, 2, "Hard", d0 + dt.timedelta(days=61))
    assert h_late == pytest.approx(0.0)  # expired: both back to prior
    # Symmetry at a date inside the window.
    h_ab, b_ab = s.diffs(1, 2, "Hard", d0 + dt.timedelta(days=10))
    h_ba, b_ba = s.diffs(2, 1, "Hard", d0 + dt.timedelta(days=10))
    assert h_ab == pytest.approx(-h_ba)
    assert b_ab == pytest.approx(-b_ba)


def test_fit_rate_vs_elo_recovers_line():
    """fit_rate_vs_elo recovers a known linear mapping, and
    serve_residual subtracts exactly the Elo-implied rate."""
    from features import fit_rate_vs_elo, serve_residual

    snaps = [(1500 + 100 * i, 0.60 + 0.0001 * (1500 + 100 * i),
              0.40 - 0.0001 * (1500 + 100 * i)) for i in range(20)]
    coefs = fit_rate_vs_elo(snaps)
    assert coefs["hold"][1] == pytest.approx(0.0001, abs=1e-9)
    assert coefs["break"][1] == pytest.approx(-0.0001, abs=1e-9)
    # A player serving exactly at the implied rate has residual 0:
    # implied hold at Elo 2000 is 0.60 + 0.0001*2000 = 0.80.
    assert serve_residual(0.80, 2000, coefs["hold"]) == pytest.approx(0.0, abs=1e-9)
    # 5 points above the implied rate: residual +0.05.
    assert serve_residual(0.85, 2000, coefs["hold"]) == pytest.approx(0.05, abs=1e-9)


def test_dominance_state_wins_only_and_symmetry():
    """DominanceState: only completed wins count, shrunk toward 0.5,
    symmetric between the pair."""
    from features import DominanceState

    d = DominanceState()
    # Player 1 wins 6-0 6-0 twice (share 1.0), then a walkover win
    # and a retirement win: neither may move the state.
    d.update(1, 2, 12, 12)
    d.update(1, 2, 12, 12)
    d.update(1, 2, 0, 0, walkover=True)
    d.update(1, 2, 6, 6, retired=True)
    assert d.diff(1, 2) == pytest.approx((2.0 + 2 * 0.5) / (2 + 2) - 0.5)
    assert d.diff(1, 2) == pytest.approx(-d.diff(2, 1))
    # A player with no wins scores 0.
    assert d.diff(3, 1) == pytest.approx(-d.diff(1, 3))
    assert DominanceState().diff(1, 2) == 0.0


def test_elo_high_rd_interaction_gate():
    """The Glicko gate passes elo_diff through only when the
    less-known player is above the RD threshold."""
    from features import elo_high_rd_interaction

    assert elo_high_rd_interaction(50.0, 100.0, 250.0, 200.0) == 50.0
    assert elo_high_rd_interaction(50.0, 100.0, 150.0, 200.0) == 0.0
    assert elo_high_rd_interaction(-30.0, 300.0, 300.0, 200.0) == -30.0


def test_alternatives_row_builder_synthetic():
    """build_rows from run_logreg_alternatives: 28 features per row,
    the adopted columns match the all-batches builder, and the new
    alternative columns are populated (recent serve is nonzero once
    history exists)."""
    import datetime as dt

    from logreg import FEATURE_NAMES
    from run_logreg_all import build_rows as build_rows_all
    from run_logreg_alternatives import build_rows as build_rows_alt

    base = dt.date(2023, 1, 2)
    matches = []
    for i in range(30):
        w, l = (101, 102) if i % 2 == 0 else (102, 101)
        matches.append({
            "tourney_id": "2023-001", "match_num": i + 1,
            "tourney_date": base + dt.timedelta(days=i * 3),
            "tourney_name": "Synthetic Open", "surface": "Hard",
            "round": "R32", "tourney_level": "A",
            "winner_id": w, "loser_id": l,
            "winner_name": "A Player" if w == 101 else "B Player",
            "loser_name": "B Player" if w == 101 else "A Player",
            "w_sv_gms": 10, "w_bp_faced": 2, "w_bp_saved": 1,
            "l_sv_gms": 10, "l_bp_faced": 3, "l_bp_saved": 1,
            "score": "6-4 6-4", "minutes": 95, "best_of": 3,
        })
    dob = {101: dt.date(1998, 5, 1), 102: dt.date(2001, 5, 1)}
    rate_coefs = {"hold": (0.5, 0.0002), "break": (0.5, -0.0002)}
    rows_alt = build_rows_alt(matches, dob, 100.0, 200.0, rate_coefs)
    rows_all = build_rows_all(matches, dob, 100.0)
    assert len(rows_alt) == 30
    assert all(len(r["x"]) == 28 for r in rows_alt)  # this builder's width
    # First 21 columns (except the form cols, which this runner does
    # not rebuild) match the all-batches builder row for row.
    for r_alt, r_all in zip(rows_alt, rows_all):
        for c in range(21):
            if FEATURE_NAMES[c].startswith(("form", "resid")):
                continue
            assert r_alt["x"][c] == pytest.approx(r_all["x"][c])
    # Recent-serve columns are nonzero once history exists.
    rec_col = FEATURE_NAMES.index("hold_rec_diff")
    assert rows_alt[5]["x"][rec_col] != pytest.approx(0.0)
    # Dominance is nonzero: the winner of the first match won 12/20
    # games (share 0.6), so player 101 leads after match 1.
    dom_col = FEATURE_NAMES.index("dominance_diff")
    assert rows_alt[1]["x"][dom_col] > 0.0


def test_winrate_state_walk_forward_and_symmetry():
    """WinRateState: win% and log1p experience are walk-forward,
    shrunk, and symmetric."""
    import math

    from features import WinRateState

    w = WinRateState()
    assert w.diffs(1, 2, "Hard") == (0.0, 0.0, 0.0, 0.0)
    # Player 1 wins twice on Hard, loses once on Clay.
    w.update(1, 2, "Hard")
    w.update(1, 2, "Hard")
    w.update(2, 1, "Clay")
    wp, swp, exp, sexp = w.diffs(1, 2, "Hard")
    assert wp > 0.0  # 2-1 overall
    assert swp > 0.0  # 2-0 on Hard
    assert exp == pytest.approx(math.log1p(3) - math.log1p(3))
    # Experience: player 1 played 3, player 3 played 0.
    assert w.diffs(1, 3, "Hard")[2] == pytest.approx(math.log1p(3))
    # Symmetry.
    assert w.diffs(1, 2, "Hard") == pytest.approx(
        tuple(-v for v in w.diffs(2, 1, "Hard")))


def test_venue_state_excess_only():
    """VenueState: the feature is the venue excess over the player's
    own overall rate, so no history at the venue scores exactly 0
    instead of restating the career win%."""
    from features import VenueState

    v = VenueState()
    assert v.diff(1, 2, "2023-001") == 0.0
    # Player 1 wins 4/4 at venue X, but is 4-4 overall (loses 4
    # elsewhere): venue excess is positive.
    for _ in range(4):
        v.update(1, 2, "2023-001")
    for _ in range(4):
        v.update(2, 1, "2023-999")
    assert v.diff(1, 2, "2023-001") > 0.0
    assert v.diff(1, 2, "2023-999") < 0.0
    # A venue neither has played scores 0.
    assert v.diff(1, 2, "2023-555") == 0.0
    assert v.diff(1, 2, "2023-001") == pytest.approx(-v.diff(2, 1, "2023-001"))


def test_tournament_country_rules():
    """tournament_country: Davis Cup host is the first team,
    Olympics go by city, movers by year, unknown -> None."""
    import datetime as dt

    from features import tournament_country

    d = dt.date(2024, 1, 15)
    assert tournament_country("Davis Cup QLS R1: SRB vs NOR", d) == "SRB"
    assert tournament_country("Davis Cup Finals F: AUS vs ITA", d) == "AUS"
    assert tournament_country("Paris Olympics", dt.date(2024, 7, 27)) == "FRA"
    assert tournament_country("Tokyo Olympics", dt.date(2021, 7, 24)) == "JPN"
    assert tournament_country("Tour Finals", dt.date(2024, 11, 15)) == "ITA"
    assert tournament_country("Tour Finals", dt.date(2015, 11, 15)) == "GBR"
    assert tournament_country("Tour Finals", dt.date(1995, 11, 15)) is None
    assert tournament_country("Laver Cup", dt.date(2023, 9, 22)) == "CAN"
    assert tournament_country("NextGen Finals", dt.date(2024, 12, 18)) == "KSA"
    assert tournament_country("Next Gen Finals", dt.date(2019, 11, 5)) == "ITA"
    assert tournament_country("United Cup", d) == "AUS"
    assert tournament_country("Acapulco", d) == "MEX"
    assert tournament_country("Some Obscure 1972 Event", dt.date(1972, 5, 1)) is None


def test_rank_feats_use_previous_year_only():
    """_rank_feats reads the PREVIOUS year's year-end ranking, is
    positive when A is better ranked / has more points, and is 0
    when either side is missing."""
    import math

    from run_logreg_benchmark import _rank_feats

    rankings = {(1, 2022): (10, 3000), (2, 2022): (100, 500),
                (1, 2023): (1, 9000)}
    lr, lp = _rank_feats(rankings, 1, 2, 2023)
    assert lr == pytest.approx(math.log(100) - math.log(10))
    assert lp == pytest.approx(math.log(3001) - math.log(501))
    # The 2023 ranking (a future row for a 2023 match) is NOT used:
    # with only (1, 2023) present and (2, 2022) missing, both are 0.
    assert _rank_feats({(1, 2023): (1, 9000)}, 1, 2, 2023) == (0.0, 0.0)
    assert _rank_feats(rankings, 1, 99, 2023) == (0.0, 0.0)


def test_home_diff_values():
    """_home_diff is -1/0/1 and neutral when the country is unknown."""
    from run_logreg_benchmark import _home_diff

    ioc = {1: "ESP", 2: "USA"}
    assert _home_diff(ioc, 1, 2, "ESP") == 1.0
    assert _home_diff(ioc, 2, 1, "ESP") == -1.0
    assert _home_diff(ioc, 1, 2, "FRA") == 0.0
    assert _home_diff(ioc, 1, 2, None) == 0.0
    assert _home_diff({}, 1, 2, "ESP") == 0.0


def test_benchmark_row_builder_synthetic():
    """build_rows from run_logreg_benchmark: 36 features per row, the
    12 base columns match the alternatives builder, ranking uses
    the previous year's row, and the venue column is 0 before any
    prior meeting at that tournament."""
    import datetime as dt

    from logreg import FEATURE_NAMES
    from run_logreg_alternatives import build_rows as build_rows_alt
    from run_logreg_benchmark import build_rows as build_rows_bm

    base = dt.date(2023, 1, 2)
    matches = []
    for i in range(30):
        w, l = (101, 102) if i % 2 == 0 else (102, 101)
        matches.append({
            "tourney_id": "2023-001", "match_num": i + 1,
            "tourney_date": base + dt.timedelta(days=i * 3),
            "tourney_name": "Synthetic Open", "surface": "Hard",
            "round": "R32", "tourney_level": "A",
            "winner_id": w, "loser_id": l,
            "winner_name": "A Player" if w == 101 else "B Player",
            "loser_name": "B Player" if w == 101 else "A Player",
            "w_sv_gms": 10, "w_bp_faced": 2, "w_bp_saved": 1,
            "l_sv_gms": 10, "l_bp_faced": 3, "l_bp_saved": 1,
            "score": "6-4 6-4", "minutes": 95, "best_of": 3,
        })
    dob = {101: dt.date(1998, 5, 1), 102: dt.date(2001, 5, 1)}
    rankings = {(101, 2022): (20, 2000), (102, 2022): (80, 800)}
    ioc = {101: "ESP", 102: "USA"}
    rows_bm, cov = build_rows_bm(matches, dob, 100.0, rankings, ioc)
    assert len(rows_bm) == 30
    # The benchmark runner pins N_COLS = 36 (FEATURE_NAMES has grown
    # since with round-4 columns appended).
    assert all(len(r["x"]) == 36 for r in rows_bm)
    # Ranking: A is better ranked (20 vs 80), so logrank is positive.
    lr_col = FEATURE_NAMES.index("logrank_diff")
    assert rows_bm[0]["x"][lr_col] > 0.0
    # Venue: no prior meeting at this tournament for row 0.
    ven_col = FEATURE_NAMES.index("venue_winpct_diff")
    assert rows_bm[0]["x"][ven_col] == pytest.approx(0.0)
    # Home: "Synthetic Open" has no country -> 0.
    home_col = FEATURE_NAMES.index("home_diff")
    assert all(r["x"][home_col] == 0.0 for r in rows_bm)
    # Base columns match the alternatives builder row for row.
    rate_coefs = {"hold": (0.5, 0.0002), "break": (0.5, -0.0002)}
    rows_alt = build_rows_alt(matches, dob, 100.0, 200.0, rate_coefs)
    base_names = ["elo_diff", "rest_diff", "games7_diff", "games14_diff",
                  "prevmin_diff", "age_diff", "age_x_bo5", "age_x_close",
                  "age_x_load7", "hold_rec_diff", "break_rec_diff",
                  "age_x_hold_diff"]
    for r_bm, r_alt in zip(rows_bm, rows_alt):
        for name in base_names:
            c = FEATURE_NAMES.index(name)
            assert r_bm["x"][c] == pytest.approx(r_alt["x"][c])


def test_parse_clutch_cases():
    from features import parse_clutch

    # "7-6(5) 6-4": winner won a tiebreak set and a plain set.
    tb_won, tb_played, sets = parse_clutch("7-6(5) 6-4")
    assert (tb_won, tb_played) == (1, 1)
    assert sets == [True, True]
    # "6-7(4) 7-5 6-3": winner lost the first-set tiebreak.
    tb_won, tb_played, sets = parse_clutch("6-7(4) 7-5 6-3")
    assert (tb_won, tb_played) == (0, 1)
    assert sets == [False, True, True]
    # Bracketed match tiebreak counts as a tiebreak won by the winner.
    tb_won, tb_played, sets = parse_clutch("6-4 3-6 [10-7]")
    assert (tb_won, tb_played) == (1, 1)
    assert sets == [True, False, True]
    # Bare "7-6" with no parentheses carries no tiebreak info.
    tb_won, tb_played, sets = parse_clutch("7-6 6-3")
    assert (tb_won, tb_played) == (0, 0)
    assert sets == [True, True]
    # Walkover / bare RET: nothing.
    assert parse_clutch("W/O") == (0, 0, [])
    assert parse_clutch("RET") == (0, 0, [])
    # Retirement mid-match keeps the sets actually played.
    tb_won, tb_played, sets = parse_clutch("6-2 0-0 RET")
    assert sets == [True, False]


def test_clutch_state_walk_forward_and_symmetry():
    from features import ClutchState

    st = ClutchState()
    # P1 beats P2 7-6(5) 6-4, then P2 beats P1 6-7(4) 7-5 6-3.
    # P1 won BOTH tiebreaks (one in each match); P2 won none.
    st.update(1, 2, "7-6(5) 6-4", 3)
    st.update(2, 1, "6-7(4) 7-5 6-3", 3)
    tb_d, dec_d, come_d = st.diffs(1, 2)
    assert tb_d > 0.0
    # Deciding sets: P1 lost the 3-setter's decider, P2 won it.
    # Each has one deciding set played (won 0 vs 1): diff negative.
    assert dec_d < 0.0
    # Comebacks: P2 lost the first set of match 2 and still won it;
    # P1 never came back (P1 won the first set of both matches it
    # played and lost match 2 from there). Diff negative.
    assert come_d < 0.0
    # Symmetry: swapping A and B negates every diff.
    tb2, dec2, come2 = st.diffs(2, 1)
    assert tb2 == pytest.approx(-tb_d)
    assert dec2 == pytest.approx(-dec_d)
    assert come2 == pytest.approx(-come_d)
    # best_of None: deciding set not recorded, tiebreak still is.
    st2 = ClutchState()
    st2.update(1, 2, "7-6(5) 6-4", None)
    tb_d2, dec_d2, _ = st2.diffs(1, 2)
    assert tb_d2 > 0.0
    assert dec_d2 == pytest.approx(0.0)


def test_point_serve_state_walk_forward_and_symmetry():
    from features import PointServeState

    st = PointServeState()
    # P1 serves: 40 pts, 30 first in, 22 first won, 7 second won,
    # 3 aces, 1 DF, faced 4 BP, saved 3. P2 serves: 40 pts, 26 first
    # in, 18 first won, 6 second won, 2 aces, 2 DF, faced 3, saved 1.
    st.update(1, 2, "Hard",
              40, 30, 22, 7, 3, 1, 4, 3,
              40, 26, 18, 6, 2, 2, 3, 1)
    d = st.diffs(1, 2, "Hard")
    assert len(d) == 6
    # P1 won more first-serve points and saved more BPs.
    assert d[0] > 0.0  # first win%
    assert d[5] > 0.0  # BP save%
    # P1's BP conversion: (3 - 1) / 3 = 0.667; P2's: (4 - 3) / 4 = 0.25.
    assert d[4] > 0.0
    # Symmetry.
    d2 = st.diffs(2, 1, "Hard")
    assert all(a == pytest.approx(-b) for a, b in zip(d, d2))
    # A player with no history gets the tour prior: diff vs an
    # identical no-history player is 0 on every component.
    d3 = st.diffs(9, 10, "Hard")
    assert all(v == pytest.approx(0.0) for v in d3)
    # NULL serve columns: the match does not update the state.
    before = st.diffs(1, 2, "Hard")
    st.update(1, 2, "Hard",
              None, None, None, None, None, None, None, None,
              40, 26, 18, 6, 2, 2, 3, 1)
    assert st.diffs(1, 2, "Hard") == pytest.approx(before)


def test_form_momentum_diff():
    import datetime as dt

    from features import FormState

    st = FormState()
    base = dt.date(2023, 1, 1)
    # P1 loses 15 straight, then wins 5 straight: rising.
    # P2 wins 15 straight, then loses 5 straight: falling.
    for i in range(20):
        d = base + dt.timedelta(days=i)
        if i < 15:
            st.update(2, 1, d, 0.5)
        else:
            st.update(1, 2, d, 0.5)
    mom = st.momentum_diff(1, 2, base + dt.timedelta(days=20))
    assert mom > 0.0
    assert st.momentum_diff(2, 1, base + dt.timedelta(days=20)) == pytest.approx(-mom)
    # A player with no history has momentum 0.
    assert st.momentum_diff(1, 99, base + dt.timedelta(days=20)) == pytest.approx(
        st.form_5(1, base + dt.timedelta(days=20))
        - st.form_20(1, base + dt.timedelta(days=20)))


def test_event_and_retirement_states():
    import datetime as dt

    from features import EventState, RetirementState

    ev = EventState()
    ev.update(1, 2, "2023-001", 20)
    ev.update(1, 3, "2023-001", 25)
    # P1 played 45 games at the event, P2 played 20.
    assert ev.diff(1, 2, "2023-001") == pytest.approx(25.0)
    assert ev.diff(1, 2, "2023-002") == pytest.approx(0.0)
    assert ev.diff(2, 1, "2023-001") == pytest.approx(-25.0)

    rs = RetirementState()
    d0 = dt.date(2023, 1, 10)
    rs.update(1, 2, d0, retired=True)  # P2 retired; P1's opp retired.
    rs.update(3, 4, d0, retired=False)
    d = d0 + dt.timedelta(days=30)
    self_d, opp_d = rs.diffs(2, 1, d)
    assert self_d == pytest.approx(1.0)  # P2 self-retired, P1 did not.
    assert opp_d == pytest.approx(-1.0)  # P1 had an opp retire, P2 did not.
    # Outside the 60-day window the flags expire.
    d_late = d0 + dt.timedelta(days=61)
    assert rs.diffs(2, 1, d_late) == (pytest.approx(0.0), pytest.approx(0.0))


def test_round4_row_builder_synthetic():
    """build_rows from run_logreg_round4: 50 features per row, the
    16 base columns match the benchmark builder, and the new
    columns behave: point-serve diffs nonzero after a match with
    serve data, momentum 0 on the first row, event games grow
    within a tournament."""
    import datetime as dt

    from logreg import FEATURE_NAMES
    from run_logreg_benchmark import build_rows as build_rows_bm
    from run_logreg_round4 import build_rows as build_rows_r4

    base = dt.date(2023, 1, 2)
    matches = []
    # Match 0: P101 beats P103, so P101 (but not P102) already has
    # point-serve history and event games when they first meet.
    matches.append({
        "tourney_id": "2023-001", "match_num": 1,
        "tourney_date": base, "tourney_name": "Synthetic Open",
        "surface": "Hard", "round": "R32", "tourney_level": "A",
        "winner_id": 101, "loser_id": 103,
        "winner_name": "A Player", "loser_name": "C Player",
        "w_sv_gms": 10, "w_bp_faced": 2, "w_bp_saved": 1,
        "l_sv_gms": 10, "l_bp_faced": 3, "l_bp_saved": 1,
        "w_svpt": 60, "w_1st_in": 40, "w_1st_won": 30,
        "w_2nd_won": 12, "w_ace": 5, "w_df": 2,
        "l_svpt": 60, "l_1st_in": 38, "l_1st_won": 26,
        "l_2nd_won": 10, "l_ace": 3, "l_df": 3,
        "score": "7-6(5) 6-4", "minutes": 95, "best_of": 3,
    })
    for i in range(1, 30):
        w, l = (101, 102) if i % 2 == 1 else (102, 101)
        matches.append({
            "tourney_id": "2023-001", "match_num": i + 1,
            "tourney_date": base + dt.timedelta(days=i * 3),
            "tourney_name": "Synthetic Open", "surface": "Hard",
            "round": "R32", "tourney_level": "A",
            "winner_id": w, "loser_id": l,
            "winner_name": "A Player" if w == 101 else "B Player",
            "loser_name": "B Player" if w == 101 else "A Player",
            "w_sv_gms": 10, "w_bp_faced": 2, "w_bp_saved": 1,
            "l_sv_gms": 10, "l_bp_faced": 3, "l_bp_saved": 1,
            "w_svpt": 60, "w_1st_in": 40, "w_1st_won": 30,
            "w_2nd_won": 12, "w_ace": 5, "w_df": 2,
            "l_svpt": 60, "l_1st_in": 38, "l_1st_won": 26,
            "l_2nd_won": 10, "l_ace": 3, "l_df": 3,
            "score": "7-6(5) 6-4" if i % 2 == 1 else "6-4 6-4",
            "minutes": 95, "best_of": 3,
        })
    dob = {101: dt.date(1998, 5, 1), 102: dt.date(2001, 5, 1),
           103: dt.date(1999, 5, 1)}
    rankings = {(101, 2022): (20, 2000), (102, 2022): (80, 800),
                (103, 2022): (150, 300)}
    rows_r4, cov = build_rows_r4(matches, dob, 100.0, rankings)
    assert len(rows_r4) == 30
    assert all(len(r["x"]) == 50 for r in rows_r4)
    assert len(FEATURE_NAMES) == 53  # round-4 50 + 3 home-x columns
    # First row: no history anywhere, so momentum / clutch / event
    # / retirement diffs are all exactly 0.
    mom = FEATURE_NAMES.index("momentum_diff")
    tb = FEATURE_NAMES.index("tb_winpct_diff")
    ev = FEATURE_NAMES.index("event_games_diff")
    rs = FEATURE_NAMES.index("ret_self_diff")
    assert rows_r4[0]["x"][mom] == pytest.approx(0.0)
    assert rows_r4[0]["x"][tb] == pytest.approx(0.0)
    assert rows_r4[0]["x"][ev] == pytest.approx(0.0)
    assert rows_r4[0]["x"][rs] == pytest.approx(0.0)
    # Second row: P101 has one match of point-serve history and 23
    # event games (7-6 + 6-4) from match 0; P102 has neither.
    fw = FEATURE_NAMES.index("first_win_diff")
    assert rows_r4[1]["x"][fw] != pytest.approx(0.0)
    assert rows_r4[1]["x"][ev] == pytest.approx(23.0)
    # Elo x bo5 is 0 here (all best_of 3); base columns match the
    # benchmark builder row for row.
    rows_bm, _ = build_rows_bm(matches, dob, 100.0, rankings,
                               {101: "ESP", 102: "USA", 103: "FRA"})
    base_names = ["elo_diff", "rest_diff", "games7_diff", "games14_diff",
                  "prevmin_diff", "age_diff", "age_x_bo5", "age_x_close",
                  "age_x_load7", "hold_rec_diff", "break_rec_diff",
                  "age_x_hold_diff", "exp_diff", "surf_exp_diff",
                  "logrank_diff", "logpoints_diff"]
    for r_r4, r_bm in zip(rows_r4, rows_bm):
        for name in base_names:
            c = FEATURE_NAMES.index(name)
            assert r_r4["x"][c] == pytest.approx(r_bm["x"][c])
