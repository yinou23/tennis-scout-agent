"""Week 3, rule-based planner, no LLM, no API key.

Purpose: prove the pipeline end-to-end before any LLM is connected.
Question -> plan (which tools, with which arguments) -> execute tools
-> answer built ONLY from tool outputs. The plan step is the ONLY
part that will be replaced by an LLM planner later. The executor,
the tools, and the answer builder stay unchanged, so switching to
an LLM is a plug-in replacement of parse_question, not a rewrite.

Supported question templates in v1 (clean input, not full sentences):
- Match: "Jannik Sinner vs Carlos Alcaraz Hard"
          "Sinner 对 Alcaraz 硬地"
          Surface is optional; without it, overall Elo is used.
- Head-to-head: "Jannik Sinner vs Carlos Alcaraz 交手"
                "Sinner 对 Alcaraz h2h"
- Player Elo: "Jannik Sinner Elo"
- Player form: "Jannik Sinner 状态" or "Jannik Sinner form"
- Player surface record: "Jannik Sinner 场地战绩"

What this planner does NOT do:
- It does not understand free-form sentences, typos, or follow-ups.
  Those are exactly what the LLM planner will add later.
- It does not invent numbers. Every number in an answer comes from
  a tennis_tools function return value, formatted by build_answer.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/planner.py "Jannik Sinner vs Carlos Alcaraz Hard"
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tennis_tools import (  # noqa: E402
    get_head_to_head,
    get_match_probability,
    get_player_elo,
    get_player_form,
    get_player_surface_record,
)

_SURFACE_KEYWORDS = [
    ("Hard", ["硬地", "hard"]),
    ("Clay", ["红土", "clay"]),
    ("Grass", ["草地", "grass"]),
    ("Carpet", ["地毯", "carpet"]),
]

_PREFIXES_TO_STRIP = [
    "who is more likely to win",
    "who will win",
    "who is the favorite",
    "预测",
    "谁会赢",
    "谁是热门",
]

_SUFFIXES_TO_STRIP = [" on", " in", " at", "在", "谁是热门", "谁会赢", "谁更可能赢", "谁赢"]


def _detect_surface(question):
    lower = question.lower()
    surface = None
    cleaned = question
    for name, keywords in _SURFACE_KEYWORDS:
        for kw in keywords:
            if kw in lower if kw.isascii() else kw in question:
                surface = name
                # Remove the keyword so it does not pollute a player name.
                cleaned = re.sub(re.escape(kw), "", cleaned, flags=re.IGNORECASE)
    return surface, cleaned


def _clean_name(raw):
    name = raw.strip().strip("?？.。!！,，").strip()
    for prefix in _PREFIXES_TO_STRIP:
        if name.lower().startswith(prefix):
            name = name[len(prefix):].strip().strip("?？.。!！,，").strip()
    for suffix in _SUFFIXES_TO_STRIP:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)].strip()
    # Strip punctuation and spaces repeatedly, so ", Sinner" becomes "Sinner",
    # not " Sinner" with a leading space.
    return name.strip().strip("?？.。!！,，").strip()


def _split_match(question):
    # Try "vs" first, then "or" / "或者", then Chinese "对".
    match = re.split(r"\s+vs\.?\s+", question, flags=re.IGNORECASE)
    if len(match) == 2:
        return _clean_name(match[0]), _clean_name(match[1])
    match = re.split(r"\s+or\s+", question, flags=re.IGNORECASE)
    if len(match) == 2 and match[0].strip() and match[1].strip():
        return _clean_name(match[0]), _clean_name(match[1])
    if "或者" in question:
        parts = question.split("或者", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return _clean_name(parts[0]), _clean_name(parts[1])
    if "对" in question:
        parts = question.split("对", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return _clean_name(parts[0]), _clean_name(parts[1])
    return None


def parse_question(question):
    """Parse a question into a plan, or None if no template matches.

    Plan format, shared with the future LLM planner:
        {"tool": "match", "player_a": ..., "player_b": ..., "surface": ...}
        {"tool": "h2h", "player_a": ..., "player_b": ...}
        {"tool": "elo", "player": ...}
        {"tool": "form", "player": ...}
        {"tool": "surface", "player": ...}
    """
    if not question or not question.strip():
        return None
    question = question.strip()
    lower = question.lower()
    surface, cleaned = _detect_surface(question)
    # Head-to-head takes priority over match when explicitly asked.
    if "交手" in question or "h2h" in lower:
        cleaned_h2h = re.sub(r"h2h", "", cleaned, flags=re.IGNORECASE).replace("交手", "")
        pair = _split_match(cleaned_h2h)
        if pair is None:
            return None
        return {"tool": "h2h", "player_a": pair[0], "player_b": pair[1]}
    pair = _split_match(cleaned)
    if pair is not None:
        return {"tool": "match", "player_a": pair[0], "player_b": pair[1], "surface": surface}
    # Single-player questions: strip the keyword, the rest is the name.
    if "elo" in lower:
        name = _clean_name(re.sub(r"elo", "", cleaned, flags=re.IGNORECASE).replace("的", ""))
        return {"tool": "elo", "player": name} if name else None
    if "状态" in question or "form" in lower or "近期" in question:
        name = _clean_name(
            re.sub(r"form", "", cleaned, flags=re.IGNORECASE)
            .replace("状态", "")
            .replace("近期", "")
            .replace("的", "")
        )
        return {"tool": "form", "player": name} if name else None
    if "场地战绩" in question or "surface record" in lower:
        name = _clean_name(
            re.sub(r"surface record", "", cleaned, flags=re.IGNORECASE)
            .replace("场地战绩", "")
            .replace("的", "")
        )
        return {"tool": "surface", "player": name} if name else None
    return None


def execute_plan(conn, plan):
    """Run the tools named in a plan. Returns tool outputs, or an error dict."""
    if plan is None:
        return {"error": "no_template_matched"}
    tool = plan["tool"]
    if tool == "match":
        prob = get_match_probability(conn, plan["player_a"], plan["player_b"], plan.get("surface"))
        if prob is None or prob.get("ambiguous"):
            return prob if prob is not None else {"error": "player_not_found"}
        form_a = get_player_form(conn, plan["player_a"])
        form_b = get_player_form(conn, plan["player_b"])
        h2h = get_head_to_head(conn, plan["player_a"], plan["player_b"])
        return {"match": prob, "form_a": form_a, "form_b": form_b, "h2h": h2h}
    if tool == "h2h":
        result = get_head_to_head(conn, plan["player_a"], plan["player_b"])
        return result if result is not None else {"error": "player_not_found"}
    if tool == "elo":
        result = get_player_elo(conn, plan["player"])
        return result if result is not None else {"error": "player_not_found"}
    if tool == "form":
        result = get_player_form(conn, plan["player"])
        return result if result is not None else {"error": "player_not_found"}
    if tool == "surface":
        result = get_player_surface_record(conn, plan["player"])
        return result if result is not None else {"error": "player_not_found"}
    return {"error": "unknown_tool"}


def detect_language(question):
    """Return "zh" if the question contains any Chinese character, else "en".

    Default is English. No LLM is involved; this is a character-range check.
    """
    if not question:
        return "en"
    for ch in question:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def build_answer(plan, outputs, question=None):
    """Build an answer ONLY from tool outputs. No new numbers are invented.

    Language follows the question: Chinese question -> Chinese answer,
    otherwise English answer. Default is English.
    """
    lang = detect_language(question)
    zh = lang == "zh"
    if outputs.get("error") == "no_template_matched":
        if zh:
            return "这个问题超出了规则版规划器的模板，请换一种问法，例如：Jannik Sinner vs Carlos Alcaraz Hard。"
        return "This question is outside the rule planner templates. Please rephrase, for example: Jannik Sinner vs Carlos Alcaraz Hard."
    if outputs.get("error") == "player_not_found":
        return "没有找到这个球员，请用全名再试一次。" if zh else "Player not found. Please try again with the full name."
    if outputs.get("ambiguous"):
        names = "、".join(c["name"] for c in outputs["candidates"]) if zh else ", ".join(c["name"] for c in outputs["candidates"])
        if zh:
            return f"这个名字有多个球员，请用全名再试一次。候选：{names}。"
        return f"This name matches multiple players. Please try again with the full name. Candidates: {names}."
    if plan["tool"] == "match":
        prob = outputs["match"]
        a = prob["player_a"]["name"]
        b = prob["player_b"]["name"]
        surface = prob["surface"] or ("综合" if zh else "Overall")
        if zh:
            lines = [
                f"{a} 对 {b}（{surface}）：热门是 {prob['favorite']}。",
                f"融合概率：{a} {prob['prob_a_calibrated']}，{b} {prob['prob_b_calibrated']}（校准后，融合前是 {prob['prob_a_blended']} 对 {prob['prob_b_blended']}）。",
                f"Elo：{a} 综合 {prob['player_a']['overall_elo']}，{surface}融合 {prob['player_a']['blended_elo']}；{b} 综合 {prob['player_b']['overall_elo']}，{surface}融合 {prob['player_b']['blended_elo']}。",
            ]
        else:
            lines = [
                f"{a} vs {b} ({surface}): Favorite is {prob['favorite']}.",
                f"Calibrated probability: {a} {prob['prob_a_calibrated']}, {b} {prob['prob_b_calibrated']} (blended: {prob['prob_a_blended']} vs {prob['prob_b_blended']}).",
                f"Elo: {a} overall {prob['player_a']['overall_elo']}, {surface} blended {prob['player_a']['blended_elo']}; {b} overall {prob['player_b']['overall_elo']}, {surface} blended {prob['player_b']['blended_elo']}.",
            ]
        h2h = outputs.get("h2h")
        if h2h and not h2h.get("ambiguous") and h2h.get("total") is not None:
            if zh:
                lines.append(f"交手：共 {h2h['total']} 场，{a} {h2h['wins_a']} 胜，{b} {h2h['wins_b']} 胜。")
            else:
                lines.append(f"Head-to-head: {h2h['total']} meetings, {a} {h2h['wins_a']} wins, {b} {h2h['wins_b']} wins.")
        if zh:
            lines.append("数据截至 2026-05-25，不是实时数据。状态、局数、退赛、交手不进概率，只做描述。")
        else:
            lines.append("Data as of 2026-05-25, not live. Form, games, retirement, and head-to-head are descriptors only, not probability inputs.")
        return "\n".join(lines)
    if plan["tool"] == "h2h":
        a = outputs["player_a"]["name"]
        b = outputs["player_b"]["name"]
        if zh:
            return f"{a} 对 {b} 交手：共 {outputs['total']} 场，{a} {outputs['wins_a']} 胜，{b} {outputs['wins_b']} 胜。数据截至 2026-05-25。"
        return f"{a} vs {b} head-to-head: {outputs['total']} meetings, {a} {outputs['wins_a']} wins, {b} {outputs['wins_b']} wins. Data as of 2026-05-25."
    if plan["tool"] == "elo":
        if zh:
            return f"{outputs['name']} 综合 Elo {outputs['overall_elo']}，{outputs['matches_played']} 场，最近比赛 {outputs['last_match_date']}。数据截至 2026-05-25。"
        return f"{outputs['name']} overall Elo {outputs['overall_elo']}, {outputs['matches_played']} matches, last match {outputs['last_match_date']}. Data as of 2026-05-25."
    if plan["tool"] == "form":
        win = outputs["win_form"]
        if zh:
            return f"{outputs['name']} 近期状态：最近 10 场胜率 {win['last10_pct']}（{win['last10_wins']} 胜 {win['last10_n']} 场），最近 52 周胜率 {win['last52_pct']}（{win['last52_wins']} 胜 {win['last52_n']} 场）。数据截至 2026-05-25。"
        return f"{outputs['name']} recent form: last 10 win rate {win['last10_pct']} ({win['last10_wins']} wins in {win['last10_n']} matches), last 52 weeks win rate {win['last52_pct']} ({win['last52_wins']} wins in {win['last52_n']} matches). Data as of 2026-05-25."
    if plan["tool"] == "surface":
        if zh:
            parts = ", ".join(f"{s} {v['wins']} 胜 {v['losses']} 负" for s, v in outputs["by_surface"].items())
            return f"{outputs['name']} 场地战绩：{parts}，总计 {outputs['total_wins']} 胜 {outputs['total_losses']} 负。数据截至 2026-05-25。"
        parts = ", ".join(f"{s} {v['wins']}W-{v['losses']}L" for s, v in outputs["by_surface"].items())
        return f"{outputs['name']} surface record: {parts}, total {outputs['total_wins']}W-{outputs['total_losses']}L. Data as of 2026-05-25."
    return "无法回答这个问题。" if zh else "Cannot answer this question."


def main(argv):
    if len(argv) < 2:
        print('Usage: python scripts/planner.py "Jannik Sinner vs Carlos Alcaraz Hard"')
        return 1
    question = argv[1]
    plan = parse_question(question)
    if plan is None:
        print(build_answer(None, {"error": "no_template_matched"}, question))
        return 1
    from db import get_connection

    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001 - missing SQLite file is a clean CLI error, not a crash
        print(f"ERROR: could not open SQLite database: {exc}")
        return 1
    with conn:
        outputs = execute_plan(conn, plan)
    print(build_answer(plan, outputs, question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
