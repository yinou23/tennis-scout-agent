"""Week 4 Step 2: templated scouting report builder, no LLM writer yet.

Purpose: prove the Week 4 report shape end to end with deterministic
code only, before any LLM writer is added. The writer here is a
template, not a model, so it cannot invent a number: every number
in the text is taken from a tennis_tools return value, or from the
small allowed parameter set listed in verify_scout_report.

Report sections, confirmed by the user 2026-10-29 00:10 EDT:
  Header: players, surface, data as of, not live
  Final probability, with correction tags on that line
  Elo comparison: overall and surface blended, per player
  Surface record: requested surface win-loss, plus career total
  Recent form: win form, games form, retirement descriptor
  Head-to-head: total and wins, descriptor only, not a probability input
  Sources and tool trace: tool names only, no extra numbers

Correction tags, confirmed 2026-10-29 00:10 EDT:
  A tag is listed only if that correction actually entered this
  match probability. Future corrections are added as extra tags on
  the same final probability line, in the same place, without
  changing the rest of the report shape.
  Calibration tag: always present, because calibrated_prob always
  runs, even when no surface is specified.
  Surface tag: present only when a surface is specified and at
  least one of the two players has at least one match on that
  surface. If no surface is specified, or neither player has a
  match on that surface, blended equals overall for both players
  and no surface tag appears.
  Surface correction applies to every surface, including Hard.
  Hard is not exempt. Example, verified on the Mac: Sinner vs
  Alcaraz Hard, overall difference 106.4 points gives 0.6485,
  surface blended difference 122.9 points gives 0.6698, calibrated
  gives 0.6378. Clay goes the other way, blended 0.5613, calibrated
  0.5492. Same formula, different direction, because the surface
  Elo values differ.

What this file does NOT do:
  It does not use an LLM, an API key, or live data.
  It does not include the last 5 head-to-head meetings with scores
  in version 1. Scores contain numbers that would need to be added
  to the verifier allowed set, and the v1 verifier is kept exact on
  purpose. Last 5 meetings are available from Tool 3 and can be
  added later with their score numbers added to the allowed set.
  It does not change the locked probability model. Surface blend
  prior=100 and calibration a=0.8, b=0.0 are the only corrections
  in the probability, exactly as locked in Week 2. Form, games,
  retirement, and head-to-head are descriptors only.
  Retirement is reported only as an observable mid-match
  retirement event, never as an injury diagnosis.

Language rule, same as the planner: Chinese question, or a
question containing any Chinese character, gives a Chinese report.
Otherwise the report is English. Default is English.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/scout_report.py "Jannik Sinner" "Carlos Alcaraz" "Hard"
    python scripts/scout_report.py "Jannik Sinner" "Carlos Alcaraz" "Clay" "Jannik Sinner 对 Carlos Alcaraz 红土谁是热门"
  The fourth argument is optional, it only controls the report
  language. The third argument, surface, may be omitted for an
  overall-only report, in which case no surface tag appears.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tennis_tools import (  # noqa: E402
    SURFACES,
    get_head_to_head,
    get_match_probability,
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)

DATA_AS_OF = "2026-05-25"
PRIOR = 100
CALIBRATION_A = 0.8

SURFACE_ZH = {
    "Hard": "硬地",
    "Clay": "红土",
    "Grass": "草地",
    "Carpet": "地毯",
    None: "综合",
}

# Numbers that may appear in report text without being a tool value.
# Window sizes, model parameters, and the data-as-of date parts.
# No other bare number is allowed in the text, by design, so the
# verifier can catch an invented number. In particular, tool names
# are written without numbers, section headings have no numbers,
# and K, initial Elo, and other Week 2 internals are not printed.
ALLOWED_PARAMETERS = {"10", "52", "100", "0.8", "2026", "05", "25"}

_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


def detect_language(question):
    """Return "zh" if any Chinese character is present, else "en".

    Same rule as planner.detect_language. Default is English.
    """
    if not question:
        return "en"
    for ch in question:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _fmt_prob(value):
    return f"{value:.4f}"


def _fmt_elo(value):
    return f"{value:.1f}"


def _fmt_pct(value):
    if value is None:
        return None
    return f"{value:.4f}"


def _surface_label(surface, zh):
    if not zh:
        return surface if surface is not None else "Overall"
    return SURFACE_ZH.get(surface, "综合")


def _check_tool_result(result, side_name):
    """Return an error string, or None if the result is usable.

    A result is unusable if it is None, ambiguous, or a player with
    no main-tour Elo row (overall_elo is None).
    """
    if result is None:
        return f"player_not_found:{side_name}"
    if result.get("ambiguous"):
        return f"ambiguous:{side_name}"
    if result.get("overall_elo") is None and "overall_elo" in result:
        return f"no_elo_row:{side_name}"
    return None


def build_scout_report(conn, player_a, player_b, surface=None, question=None):
    """Build a templated scouting report from tool outputs only.

    Returns a dict with "text", "values", "tags", "surface_applied",
    and the raw tool outputs under "tools", or a dict with "error"
    if any tool result is unusable. No number is invented here:
    "values" is the exact list of number strings that appear in
    "text", apart from ALLOWED_PARAMETERS and the date parts.
    """
    zh = detect_language(question) == "zh"
    if surface is not None and surface not in SURFACES:
        surface = None

    prob = get_match_probability(conn, player_a, player_b, surface)
    err = _check_tool_result(prob, "match")
    if err is not None:
        return {"error": err, "candidates": prob.get("candidates") if isinstance(prob, dict) else None}
    if "player_a" not in prob or "player_b" not in prob:
        return {"error": "no_elo_row:match"}

    elo_a = get_player_elo(conn, player_a)
    elo_b = get_player_elo(conn, player_b)
    form_a = get_player_form(conn, player_a)
    form_b = get_player_form(conn, player_b)
    record_a = get_player_surface_record(conn, player_a)
    record_b = get_player_surface_record(conn, player_b)
    h2h = get_head_to_head(conn, player_a, player_b)

    for label, result in (
        ("elo_a", elo_a),
        ("elo_b", elo_b),
        ("form_a", form_a),
        ("form_b", form_b),
        ("record_a", record_a),
        ("record_b", record_b),
        ("h2h", h2h),
    ):
        if result is None:
            return {"error": f"player_not_found:{label}"}
        if result.get("ambiguous"):
            return {"error": f"ambiguous:{label}", "candidates": result.get("candidates")}

    name_a = prob["player_a"]["name"]
    name_b = prob["player_b"]["name"]

    # Surface tag rule, confirmed 2026-10-29: present only when a
    # surface is specified and at least one player has a match on it.
    surface_applied = bool(
        surface is not None
        and (
            prob["player_a"]["surface_matches"] > 0
            or prob["player_b"]["surface_matches"] > 0
        )
    )

    surface_label = _surface_label(surface, zh)
    if surface_applied:
        if zh:
            surface_tag = f"[场地修正：{surface_label}融合 prior=100]"
        else:
            surface_tag = f"[Surface blend: {surface}, prior=100]"
    else:
        surface_tag = None
    if zh:
        calibration_tag = "[校准：a=0.8]"
    else:
        calibration_tag = "[Calibration: a=0.8]"
    tags = [t for t in (surface_tag, calibration_tag) if t is not None]

    values = []

    def add_value(text):
        if text is not None:
            values.append(str(text))
        return text

    # Final probability values.
    prob_a_cal = add_value(_fmt_prob(prob["prob_a_calibrated"]))
    prob_b_cal = add_value(_fmt_prob(prob["prob_b_calibrated"]))
    prob_a_blended = add_value(_fmt_prob(prob["prob_a_blended"]))
    prob_b_blended = add_value(_fmt_prob(prob["prob_b_blended"]))
    favorite = prob["favorite"]

    # Elo values, per player. Blended Elo is added to the verifier
    # values only when it is actually printed, which is when a
    # surface is specified. In an overall-only report, blended
    # equals overall, and only overall is printed.
    elo_text = {}
    for side, p, elo in (("a", prob["player_a"], elo_a), ("b", prob["player_b"], elo_b)):
        overall = add_value(_fmt_elo(p["overall_elo"]))
        if surface is not None:
            blended = add_value(_fmt_elo(p["blended_elo"]))
        else:
            blended = _fmt_elo(p["blended_elo"])
        if surface is not None and p["surface_elo"] is not None:
            surf_elo = add_value(_fmt_elo(p["surface_elo"]))
        else:
            surf_elo = None
        surf_n = add_value(str(p["surface_matches"])) if surface is not None else None
        matches = add_value(str(elo["matches_played"]))
        elo_text[side] = {
            "overall": overall,
            "blended": blended,
            "surface_elo": surf_elo,
            "surface_matches": surf_n,
            "matches_played": matches,
            "last_match_date": elo["last_match_date"],
        }

    # Surface record values, requested surface only, plus totals.
    record_text = {}
    for side, record in (("a", record_a), ("b", record_b)):
        entry = record["by_surface"].get(surface) if surface is not None else None
        if entry is not None:
            wins = add_value(str(entry["wins"]))
            losses = add_value(str(entry["losses"]))
            total = add_value(str(entry["total"]))
            win_pct = add_value(_fmt_pct(entry["win_pct"]))
        else:
            # The text for this branch writes "0 matches" for the
            # requested surface, so 0 must be a verifier value here,
            # otherwise the verifier would flag it as invented.
            if surface is not None:
                add_value("0")
            wins = losses = total = win_pct = None
        total_matches = add_value(str(record["total_matches"]))
        total_wins = add_value(str(record["total_wins"]))
        total_losses = add_value(str(record["total_losses"]))
        record_text[side] = {
            "entry": entry,
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_pct": win_pct,
            "total_matches": total_matches,
            "total_wins": total_wins,
            "total_losses": total_losses,
        }

    # Form values, per player. Pct values are added only when present.
    def form_lines(name, form):
        win = form["win_form"]
        games = form["games_form"]
        ret = form["retirement"]
        win10 = _fmt_pct(win["last10_pct"])
        win52 = _fmt_pct(win["last52_pct"])
        if win10 is not None:
            add_value(win10)
        if win52 is not None:
            add_value(win52)
        add_value(str(win["last10_wins"]))
        add_value(str(win["last10_n"]))
        add_value(str(win["last52_wins"]))
        add_value(str(win["last52_n"]))
        games10 = _fmt_pct(games["last10_games_pct"])
        games52 = _fmt_pct(games["last52_games_pct"])
        if games10 is not None:
            add_value(games10)
        if games52 is not None:
            add_value(games52)
        add_value(str(games["last10_games_won"]))
        add_value(str(games["last10_games_total"]))
        add_value(str(games["last52_games_won"]))
        add_value(str(games["last52_games_total"]))
        if ret["last_retirement_date"] is not None:
            # The retirement date is printed in full, so its
            # year, month, and day parts must be verifier values,
            # otherwise the verifier would flag them as invented.
            for part in str(ret["last_retirement_date"]).split("-"):
                add_value(part)
        if ret["days_since_last_retirement"] is not None:
            add_value(str(ret["days_since_last_retirement"]))
        if ret["matches_since_last_retirement"] is not None:
            add_value(str(ret["matches_since_last_retirement"]))
        add_value(str(ret["retirements_last10"]))
        add_value(str(ret["retirements_last52"]))
        add_value(str(ret["retirements_total"]))
        return {
            "win": win,
            "win10": win10,
            "win52": win52,
            "games": games,
            "games10": games10,
            "games52": games52,
            "retirement": ret,
        }

    form_text_a = form_lines(name_a, form_a)
    form_text_b = form_lines(name_b, form_b)

    # Head-to-head values.
    add_value(str(h2h["total"]))
    add_value(str(h2h["wins_a"]))
    add_value(str(h2h["wins_b"]))
    h2h_surface_entry = None
    if surface is not None:
        h2h_surface_entry = h2h["by_surface"].get(surface)
        if h2h_surface_entry is not None:
            add_value(str(h2h_surface_entry["total"]))
            add_value(str(h2h_surface_entry["wins_a"]))
            add_value(str(h2h_surface_entry["wins_b"]))

    # Deduplicate values while keeping order, for the verifier.
    seen = set()
    unique_values = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)

    # Build the text. Every number below is either in unique_values
    # or in ALLOWED_PARAMETERS, or part of the data-as-of date.
    # No other number is written here, by design.
    lines = []
    if zh:
        lines.append(f"{name_a} 对 {name_b}（{surface_label}）球探报告")
        lines.append(f"数据截至 {DATA_AS_OF}，不是实时数据。")
        tag_text = " ".join(tags)
        lines.append(f"最终概率（校准后）：{name_a} {prob_a_cal}，{name_b} {prob_b_cal} {tag_text}")
        lines.append(f"融合概率（校准前）：{name_a} {prob_a_blended}，{name_b} {prob_b_blended}。热门是 {favorite}。")
        lines.append("Elo 对比：")
        for name, side in ((name_a, "a"), (name_b, "b")):
            e = elo_text[side]
            p = prob[f"player_{side}"]
            if surface is not None and e["surface_elo"] is not None:
                lines.append(
                    f"{name}：综合 {e['overall']}，{surface_label} Elo {e['surface_elo']}（{e['surface_matches']} 场），{surface_label}融合 {e['blended']}，共 {e['matches_played']} 场。"
                )
            elif surface is not None:
                lines.append(
                    f"{name}：综合 {e['overall']}，{surface_label} 0 场，{surface_label}融合等于综合 {e['blended']}，共 {e['matches_played']} 场。"
                )
            else:
                lines.append(f"{name}：综合 {e['overall']}，共 {e['matches_played']} 场。")
        lines.append("场地战绩：")
        for name, side in ((name_a, "a"), (name_b, "b")):
            r = record_text[side]
            if surface is not None and r["entry"] is not None:
                lines.append(
                    f"{name} {surface_label}：{r['wins']} 胜 {r['losses']} 负，共 {r['total']} 场，胜率 {r['win_pct']}；生涯共 {r['total_matches']} 场，{r['total_wins']} 胜 {r['total_losses']} 负。"
                )
            elif surface is not None:
                lines.append(
                    f"{name} {surface_label}：0 场；生涯共 {r['total_matches']} 场，{r['total_wins']} 胜 {r['total_losses']} 负。"
                )
            else:
                lines.append(
                    f"{name}：生涯共 {r['total_matches']} 场，{r['total_wins']} 胜 {r['total_losses']} 负。"
                )
        lines.append("近期状态（只做描述，不进概率）：")
        for name, ft in ((name_a, form_text_a), (name_b, form_text_b)):
            win = ft["win"]
            games = ft["games"]
            ret = ft["retirement"]
            win10_text = f"最近 10 场胜率 {ft['win10']}（{win['last10_wins']} 胜 {win['last10_n']} 场）" if ft["win10"] is not None else "最近 10 场无胜率"
            win52_text = f"最近 52 周胜率 {ft['win52']}（{win['last52_wins']} 胜 {win['last52_n']} 场）" if ft["win52"] is not None else "最近 52 周无胜率"
            games10_text = f"局数最近 10 场 {ft['games10']}（{games['last10_games_won']} 局，共 {games['last10_games_total']} 局）" if ft["games10"] is not None else "局数最近 10 场无完整比赛"
            games52_text = f"局数最近 52 周 {ft['games52']}（{games['last52_games_won']} 局，共 {games['last52_games_total']} 局）" if ft["games52"] is not None else "局数最近 52 周无完整比赛"
            if ret["last_retirement_date"] is not None:
                ret_text = f"最近一次中途退赛 {ret['last_retirement_date']}，距数据截至日 {ret['days_since_last_retirement']} 天，之后 {ret['matches_since_last_retirement']} 场"
            else:
                ret_text = "记录里没有中途退赛"
            lines.append(
                f"{name}：{win10_text}，{win52_text}；{games10_text}，{games52_text}；{ret_text}；最近 10 场退赛 {ret['retirements_last10']} 次，最近 52 周退赛 {ret['retirements_last52']} 次，生涯退赛 {ret['retirements_total']} 次。"
            )
        lines.append("交手（只做描述，不进概率）：")
        if surface is not None and h2h_surface_entry is not None:
            lines.append(
                f"共 {h2h['total']} 场，{name_a} {h2h['wins_a']} 胜，{name_b} {h2h['wins_b']} 胜；{surface_label}共 {h2h_surface_entry['total']} 场，{name_a} {h2h_surface_entry['wins_a']} 胜，{name_b} {h2h_surface_entry['wins_b']} 胜。"
            )
        else:
            lines.append(f"共 {h2h['total']} 场，{name_a} {h2h['wins_a']} 胜，{name_b} {h2h['wins_b']} 胜。")
        lines.append("来源与工具：球员 Elo、近期状态、交手记录、场地战绩、比赛概率，全部来自数据库查询，不是模型凭空写出。状态、局数、退赛、交手不进概率。")
    else:
        lines.append(f"Scouting report: {name_a} vs {name_b} ({surface_label})")
        lines.append(f"Data as of {DATA_AS_OF}, not live.")
        tag_text = " ".join(tags)
        lines.append(f"Final probability (calibrated): {name_a} {prob_a_cal}, {name_b} {prob_b_cal} {tag_text}")
        lines.append(f"Blended probability (before calibration): {name_a} {prob_a_blended}, {name_b} {prob_b_blended}. Favorite is {favorite}.")
        lines.append("Elo comparison:")
        for name, side in ((name_a, "a"), (name_b, "b")):
            e = elo_text[side]
            if surface is not None and e["surface_elo"] is not None:
                lines.append(
                    f"{name}: overall {e['overall']}, {surface_label} Elo {e['surface_elo']} ({e['surface_matches']} matches), {surface_label} blended {e['blended']}, {e['matches_played']} matches total."
                )
            elif surface is not None:
                lines.append(
                    f"{name}: overall {e['overall']}, {surface_label} 0 matches, {surface_label} blended equals overall {e['blended']}, {e['matches_played']} matches total."
                )
            else:
                lines.append(f"{name}: overall {e['overall']}, {e['matches_played']} matches total.")
        lines.append("Surface record:")
        for name, side in ((name_a, "a"), (name_b, "b")):
            r = record_text[side]
            if surface is not None and r["entry"] is not None:
                lines.append(
                    f"{name} {surface_label}: {r['wins']} wins, {r['losses']} losses, {r['total']} matches, win rate {r['win_pct']}; career {r['total_matches']} matches, {r['total_wins']} wins, {r['total_losses']} losses."
                )
            elif surface is not None:
                lines.append(
                    f"{name} {surface_label}: 0 matches; career {r['total_matches']} matches, {r['total_wins']} wins, {r['total_losses']} losses."
                )
            else:
                lines.append(
                    f"{name}: career {r['total_matches']} matches, {r['total_wins']} wins, {r['total_losses']} losses."
                )
        lines.append("Recent form (descriptors only, not probability inputs):")
        for name, ft in ((name_a, form_text_a), (name_b, form_text_b)):
            win = ft["win"]
            games = ft["games"]
            ret = ft["retirement"]
            win10_text = f"last 10 win rate {ft['win10']} ({win['last10_wins']} wins in {win['last10_n']} matches)" if ft["win10"] is not None else "no last 10 win rate"
            win52_text = f"last 52 weeks win rate {ft['win52']} ({win['last52_wins']} wins in {win['last52_n']} matches)" if ft["win52"] is not None else "no last 52 weeks win rate"
            games10_text = f"games last 10 {ft['games10']} ({games['last10_games_won']} games won, {games['last10_games_total']} total)" if ft["games10"] is not None else "no complete games last 10"
            games52_text = f"games last 52 weeks {ft['games52']} ({games['last52_games_won']} games won, {games['last52_games_total']} total)" if ft["games52"] is not None else "no complete games last 52 weeks"
            if ret["last_retirement_date"] is not None:
                ret_text = f"last mid-match retirement {ret['last_retirement_date']}, {ret['days_since_last_retirement']} days before data as of, {ret['matches_since_last_retirement']} matches since"
            else:
                ret_text = "no mid-match retirement in record"
            lines.append(
                f"{name}: {win10_text}, {win52_text}; {games10_text}, {games52_text}; {ret_text}; retirements last 10: {ret['retirements_last10']}, last 52 weeks: {ret['retirements_last52']}, total: {ret['retirements_total']}."
            )
        lines.append("Head-to-head (descriptor only, not a probability input):")
        if surface is not None and h2h_surface_entry is not None:
            lines.append(
                f"{h2h['total']} meetings, {name_a} {h2h['wins_a']} wins, {name_b} {h2h['wins_b']} wins; {surface_label} {h2h_surface_entry['total']} meetings, {name_a} {h2h_surface_entry['wins_a']} wins, {name_b} {h2h_surface_entry['wins_b']} wins."
            )
        else:
            lines.append(f"{h2h['total']} meetings, {name_a} {h2h['wins_a']} wins, {name_b} {h2h['wins_b']} wins.")
        lines.append("Sources and tools: player Elo, recent form, head-to-head, surface record, match probability, all from database queries, not model text. Form, games, retirement, and head-to-head are descriptors only.")

    text = "\n".join(lines)
    return {
        "text": text,
        "values": unique_values,
        "tags": tags,
        "surface": surface,
        "surface_applied": surface_applied,
        "favorite": favorite,
        "prob_a_calibrated": prob["prob_a_calibrated"],
        "prob_b_calibrated": prob["prob_b_calibrated"],
        "prob_a_blended": prob["prob_a_blended"],
        "prob_b_blended": prob["prob_b_blended"],
        "tools": {
            "match": prob,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "form_a": form_a,
            "form_b": form_b,
            "record_a": record_a,
            "record_b": record_b,
            "h2h": h2h,
        },
        "data_as_of": DATA_AS_OF,
    }


def verify_scout_report(report):
    """Verify a templated report. This is the Week 4 Step 2 verifier.

    Checks, in order:
      1. Every value in report["values"] appears in report["text"].
      2. Every number in report["text"] is either in
         report["values"], or in ALLOWED_PARAMETERS, or part of the
         data-as-of date. Any other number is an invented number.
      3. Tag accuracy: calibration tag must be present. Surface tag
         must be present exactly when report["surface_applied"] is
         true, and must be absent otherwise.
    Returns {"ok": bool, "missing_values": [...],
             "unexpected_numbers": [...], "tag_error": str or None}.
    This verifier is exact for a templated writer. A future LLM
    writer will need a stronger verifier, because LLM text can
    contain numbers in free sentences, not only in template slots.
    """
    if "text" not in report or "values" not in report:
        return {
            "ok": False,
            "missing_values": [],
            "unexpected_numbers": [],
            "tag_error": "report missing text or values",
        }
    text = report["text"]
    missing = [v for v in report["values"] if v not in text]
    allowed = set(report["values"]) | ALLOWED_PARAMETERS
    found = _NUMBER_RE.findall(text)
    unexpected = []
    for num in found:
        if num not in allowed and num not in unexpected:
            unexpected.append(num)
    tag_error = None
    tags = report.get("tags", [])
    calibration_present = any("Calibration" in t or "校准" in t for t in tags)
    # Check the text itself, not only the tags list, so a writer
    # that drops a tag from the text is caught too.
    calibration_in_text = ("Calibration" in text) or ("校准" in text)
    surface_in_text = ("Surface blend" in text) or ("场地修正" in text)
    if not calibration_present or not calibration_in_text:
        tag_error = "calibration tag missing"
    elif bool(report.get("surface_applied")) != surface_in_text:
        tag_error = "surface tag does not match surface_applied"
    ok = not missing and not unexpected and tag_error is None
    return {
        "ok": ok,
        "missing_values": missing,
        "unexpected_numbers": unexpected,
        "tag_error": tag_error,
    }


def main(argv):
    if len(argv) < 3:
        print('Usage: python scripts/scout_report.py "Jannik Sinner" "Carlos Alcaraz" "Hard"')
        print('Usage: python scripts/scout_report.py "Jannik Sinner" "Carlos Alcaraz"')
        return 1
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a")
        return 1
    import psycopg

    player_a = argv[1]
    player_b = argv[2]
    surface = argv[3] if len(argv) >= 4 else None
    question = argv[4] if len(argv) >= 5 else f"{player_a} vs {player_b} {surface or ''}".strip()
    with psycopg.connect(database_url) as conn:
        report = build_scout_report(conn, player_a, player_b, surface, question)
    if "error" in report:
        print(f"ERROR: {report['error']}")
        if report.get("candidates"):
            for cand in report["candidates"]:
                print(f"  {cand['player_id']}: {cand['name']}")
        return 1
    check = verify_scout_report(report)
    print(report["text"])
    print()
    print(f"Verifier: ok={check['ok']}, missing={check['missing_values']}, unexpected={check['unexpected_numbers']}, tag_error={check['tag_error']}")
    if not check["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
