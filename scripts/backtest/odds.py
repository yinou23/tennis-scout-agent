"""Odds readers and de-vigging. Stdlib only (no pandas in the venv).

Two sources:

1. tennis-data.co.uk (PRIMARY, specified by the task): one Excel
   file per season, https://www.tennis-data.co.uk/{year}/{year}.xlsx
   (ATP), columns include Winner, Loser, Tournament, Round and the
   bookmaker columns B365W/B365L (Bet365) and PSW/PSL (Pinnacle).
   STATUS 2026-11-18: validated against real files 2022-2025
   (via a friend in Europe; the site Cloudflare-blocks US IPs).
   First real-file run exposed a real bug in names.name_key
   (multi-word surnames with an initial); fixed there, with a
   regression test. See README for the measured match rates.

2. Valuebetennis (FALLBACK, Pinnacle closing only, 2021+):
   https://www.valuebetennis.com/datasets/valuebetennis-matchs-{year}.csv
   semicolon-separated, CC BY 4.0. Columns: date, tournoi,
   categorie, genre, surface, joueur1, joueur2,
   cote1_ouverture, cote2_ouverture, cote1_cloture,
   cote2_cloture, vainqueur_id, score, duree_min.
   Coverage: all levels (ITF/Challenger/main tour) and both
   tours; alignment filters to ATP main-tour categories and
   matches only Sackmann main-tour rows. Closing odds are
   present for the 2023-2024 main-tour rows checked
   (1887/1913 and 1796/1835 in the category pre-filter).
   This source can validate the pipeline on 2023-2024, but it
   is NOT a substitute for tennis-data 2001-2024 (no Bet365,
   nothing before 2021). Numbers from it are labelled as such.

OddsRow: one odds row, with the two players' names, the odds
date (actual match date where the source has one), the
tournament name, and a dict of bookmaker -> (odds_a, odds_b)
in the same player order as the names.
"""

import csv
import datetime as _dt
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class OddsRow:
    source: str
    year: int
    date: object  # datetime.date or None
    tournament: str
    round: str
    player_a: str
    player_b: str
    odds: dict = field(default_factory=dict)  # bookmaker -> (odds_a, odds_b)


def _header_map(headers):
    """Map lowercased, stripped header names to column indexes."""
    out = {}
    for i, h in enumerate(headers):
        if h is None:
            continue
        out[str(h).strip().lower()] = i
    return out


def _pick(row, hmap, *names):
    for name in names:
        idx = hmap.get(name)
        if idx is not None and idx < len(row):
            val = row[idx]
            if val is not None and str(val).strip() != "":
                return val
    return None


def _to_float(val):
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def _excel_serial_to_date(val):
    """Excel serial day number -> date (1900 system, with the 1900 leap bug)."""
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if num < 20000 or num > 80000:
        return None
    base = _dt.date(1899, 12, 30)
    return base + _dt.timedelta(days=int(num))


def _parse_date(val):
    if val is None:
        return None
    if isinstance(val, (_dt.date, _dt.datetime)):
        return val.date() if isinstance(val, _dt.datetime) else val
    text = str(val).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return _excel_serial_to_date(text)


# ---------------------------------------------------------------------------
# tennis-data.co.uk xlsx (stdlib OOXML reader)
# ---------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_xlsx_rows(path):
    """Yield rows (lists of values) from the first sheet of an .xlsx file."""
    with zipfile.ZipFile(path) as zf:
        try:
            shared = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")) for si in shared]
        except KeyError:
            strings = []
        sheet_names = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = sheet_names.find("m:sheets/m:sheet", _NS)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet is not None:
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            for rel in rels:
                if rel.get("Id") == rid:
                    sheet_path = "xl/" + rel.get("Target").lstrip("/")
                    break
        root = ET.fromstring(zf.read(sheet_path))
        for row_el in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
            cells = {}
            max_col = -1
            for c in row_el:
                ref = c.get("r", "")
                m = re.match(r"([A-Z]+)", ref)
                if not m:
                    continue
                col = 0
                for ch in m.group(1):
                    col = col * 26 + (ord(ch) - 64)
                col -= 1
                max_col = max(max_col, col)
                ctype = c.get("t")
                v = c.find("m:v", _NS)
                is_el = c.find("m:is", _NS)
                if ctype == "s" and v is not None:
                    cells[col] = strings[int(v.text)] if v.text and int(v.text) < len(strings) else ""
                elif is_el is not None:
                    cells[col] = "".join(t.text or "" for t in is_el.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                elif v is not None:
                    cells[col] = v.text
                else:
                    cells[col] = None
            if max_col < 0:
                continue
            yield [cells.get(i) for i in range(max_col + 1)]


def read_tennis_data_xlsx(path, year):
    """Read one tennis-data.co.uk season file into OddsRows.

    Header-driven: the Winner/Loser/Tournament/Round/Date and
    B365W/B365L/PSW/PSL columns are located by header name, so a
    column reordering does not silently misalign odds. Raises
    ValueError if the required columns are absent.
    """
    rows = list(_read_xlsx_rows(path))
    if not rows:
        raise ValueError(f"{path}: empty workbook")
    header_idx = None
    for i, row in enumerate(rows[:5]):
        vals = [str(v).strip().lower() if v is not None else "" for v in row]
        if "winner" in vals and "loser" in vals:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path}: no header row with Winner/Loser in first 5 rows")
    hmap = _header_map(rows[header_idx])
    for needed in ("winner", "loser"):
        if needed not in hmap:
            raise ValueError(f"{path}: missing column {needed}")
    out = []
    for row in rows[header_idx + 1:]:
        winner = _pick(row, hmap, "winner")
        loser = _pick(row, hmap, "loser")
        if not winner or not loser:
            continue
        odds = {}
        b365w, b365l = _to_float(_pick(row, hmap, "b365w")), _to_float(_pick(row, hmap, "b365l"))
        if b365w and b365l:
            odds["bet365"] = (b365w, b365l)
        psw, psl = _to_float(_pick(row, hmap, "psw")), _to_float(_pick(row, hmap, "psl"))
        if psw and psl:
            odds["pinnacle"] = (psw, psl)
        if not odds:
            continue
        out.append(OddsRow(
            source="tennis-data.co.uk",
            year=year,
            date=_parse_date(_pick(row, hmap, "date")),
            tournament=str(_pick(row, hmap, "tournament") or ""),
            round=str(_pick(row, hmap, "round") or ""),
            player_a=str(winner),
            player_b=str(loser),
            odds=odds,
        ))
    return out


# ---------------------------------------------------------------------------
# Valuebetennis CSV (fallback, Pinnacle closing only, 2021+)
# ---------------------------------------------------------------------------

_VB_MAIN_CATEGORIES = {"Main tour", "Grand Chelem", "Masters", "Coupe Davis / BJK Cup"}


def read_valuebetennis_csv(path, year, which="closing"):
    """Read one Valuebetennis season CSV into OddsRows (ATP only).

    Keeps only genre=atp rows in main-tour categories; Challenger
    and ITF rows are dropped before alignment (they can never
    match a Sackmann main-tour row). Player order is joueur1 /
    joueur2 as listed; the winner is not used for ordering.

    which="closing" reads cote1/2_cloture (default, existing
    behaviour); which="opening" reads cote1/2_ouverture instead.
    Opening odds are sparser than closing odds in this source.
    """
    if which == "opening":
        col1, col2 = "cote1_ouverture", "cote2_ouverture"
    elif which == "closing":
        col1, col2 = "cote1_cloture", "cote2_cloture"
    else:
        raise ValueError(f"which must be 'opening' or 'closing', got {which!r}")
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("genre") != "atp":
                continue
            if row.get("categorie") not in _VB_MAIN_CATEGORIES:
                continue
            p1, p2 = row.get("joueur1"), row.get("joueur2")
            o1, o2 = _to_float(row.get(col1)), _to_float(row.get(col2))
            if not p1 or not p2 or not o1 or not o2:
                continue
            date = _parse_date((row.get("date") or "").split(" ")[0])
            out.append(OddsRow(
                source="valuebetennis",
                year=year,
                date=date,
                tournament=row.get("tournoi") or "",
                round="",
                player_a=p1,
                player_b=p2,
                odds={"pinnacle": (o1, o2)},
            ))
    return out
