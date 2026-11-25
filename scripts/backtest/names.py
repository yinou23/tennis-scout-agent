"""Player-name normalisation for odds alignment.

tennis-data.co.uk and Valuebetennis write names differently from
Sackmann (full "First Last" in atp_matches.winner_name /
loser_name). Problems handled here:

- Accents: Djokovic vs Đoković, Auger-Aliassime vs Auger Aliassime.
  NFD-decompose and drop combining marks, lowercase.
- Hyphens / apostrophes / periods: "Auger-Aliassime" and
  "Auger Aliassime" normalise the same; "O'Brien" -> "obrien".
- Initials: tennis-data historically writes "Nadal R" style
  (surname + first initial). Sackmann writes "Rafael Nadal".
  The match key is therefore (surname, first-initial), with the
  full normalised name kept for tie-breaking.
- Multi-word surnames with an initial: tennis-data writes
  "O Connell C.", "De Minaur A.", "Van Assche L.",
  "Auger Aliassime F.", "Van De Zandschulp B.". The surname is
  the token before the initial, not the initial itself. Found
  2026-11-17 on the first real tennis-data files: the old code
  fell through to (initial, first-token-initial), so every match
  involving such a player came back "no_pair" and the 2022-2024
  match rate sat at ~72% instead of ~95%. The single-letter
  leading particle is the exception: "O Connell C." must key as
  ("oconnell", "c") because Sackmann writes the name as one
  token, "Christopher Oconnell".
- Married-name changes (e.g. WTA) are not handled here; this
  package is ATP only and the alias table in align.py is the
  place to add known cases if they appear in the unmatched log.

The key is deliberately lossy (surname + initial): two different
players can share it (e.g. two unrelated "Smith J"s). That is why
alignment also requires the date window and, for tennis-data,
the tournament/round, and why ambiguous pairs are reported
unmatched instead of forced.
"""

import re
import unicodedata

# Letters NFD cannot decompose (they are letters in their own right,
# not a base letter plus a combining mark). Map them before stripping
# accents so "Đoković" and "Djokovic" normalise the same.
_TRANSLITERATE = str.maketrans({
    "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ø": "o", "Ø": "o",
    "æ": "ae", "Æ": "ae", "ß": "ss", "þ": "th", "Þ": "th", "ð": "d", "Ð": "d",
})


def normalize_name(name):
    """Lowercase, strip accents, drop punctuation, collapse spaces.

    Apostrophes are deleted, not turned into spaces: "O'Connell"
    and Sackmann's "Oconnell" must normalise the same ("oconnell"),
    not to "o connell" (surname "connell").
    """
    if name is None:
        return ""
    text = str(name).translate(_TRANSLITERATE)
    text = text.replace("'", "").replace("’", "").replace("`", "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_key(name):
    """(surname, first-initial) key for a name in any of the formats.

    Accepts "Rafael Nadal", "Nadal R", "Nadal, Rafael", "NADAL R.".
    Surname is the last token unless the name is "Surname Initial"
    (two tokens, second token a single letter) or "Surname, First".
    Multi-word surname + initial ("O Connell C.", "De Minaur A.",
    "Van De Zandschulp B."): surname is the token before the
    initial; a single-letter leading particle ("O" in "O Connell")
    is prepended, since Sackmann writes that name as one token
    ("Christopher Oconnell").
    """
    norm = normalize_name(name)
    if not norm:
        return ("", "")
    if "," in str(name):
        parts = [p.strip() for p in str(name).split(",", 1)]
        surname = normalize_name(parts[0]).split()
        first = normalize_name(parts[1]).split()
        return (surname[-1] if surname else "", first[0][0] if first and first[0] else "")
    tokens = norm.split()
    if len(tokens) >= 3 and len(tokens[-1]) == 1:
        if len(tokens[-2]) == 1:
            # Surname + two initials, e.g. tennis-data "Kwon S.W.".
            return (tokens[0], tokens[1])
        surname = tokens[-2]
        if len(tokens[0]) == 1:
            surname = tokens[0] + surname
        return (surname, tokens[-1])
    if len(tokens) == 2 and len(tokens[1]) == 1:
        return (tokens[0], tokens[1])
    if len(tokens) == 1:
        return (tokens[0], "")
    return (tokens[-1], tokens[0][0])


def pair_key(name_a, name_b):
    """Order-independent key for a pair of players."""
    return tuple(sorted([name_key(name_a), name_key(name_b)]))
