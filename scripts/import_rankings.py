"""Import Sackmann ATP rankings into PostgreSQL.

Usage, from the project root ~/TSA, after activating the venv:
    source .venv/bin/activate
    set -a; source .env; set +a
    python scripts/import_rankings.py

Input files, all with header ranking_date,rank,player,points:
    atp_rankings_70s.csv, 80s, 90s, 00s, 10s, 20s, current
Note: the CSV column named "player" holds a player_id, not a name.
It joins to atp_players.player_id.

The script drops and recreates atp_rankings, so rerunning it gives
a clean import. It reads DATABASE_URL from the environment.
"""

import csv
import os
import sys
from datetime import date
from pathlib import Path

import psycopg


def to_int(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def to_date(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or len(value) != 8:
        return None
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("TSA_DATA_DIR", project_root / "data" / "raw" / "tennis_atp"))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set. Run: set -a; source .env; set +a", file=sys.stderr)
        return 1
    files = sorted(data_dir.glob("atp_rankings_*.csv"))
    if not files:
        print(f"ERROR: no ranking files found in {data_dir}", file=sys.stderr)
        return 1
    print(f"Data dir: {data_dir}")
    print(f"Ranking files: {len(files)}")
    for path in files:
        print(f"  {path.name}")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS atp_rankings CASCADE")
            cur.execute(
                """
                CREATE TABLE atp_rankings (
                    ranking_date DATE NOT NULL,
                    ranking      INTEGER,
                    player_id    INTEGER NOT NULL,
                    points       INTEGER,
                    PRIMARY KEY (ranking_date, player_id)
                )
                """
            )
            cur.execute("CREATE INDEX idx_rankings_player ON atp_rankings (player_id)")
            cur.execute("CREATE INDEX idx_rankings_date ON atp_rankings (ranking_date)")

            total = 0
            sql = (
                "INSERT INTO atp_rankings VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (ranking_date, player_id) DO NOTHING"
            )
            for path in files:
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    # Verify header, do not guess it.
                    expected = ["ranking_date", "rank", "player", "points"]
                    if reader.fieldnames != expected:
                        print(f"ERROR: {path.name} header is {reader.fieldnames}, expected {expected}", file=sys.stderr)
                        return 1
                    batch = []
                    count = 0
                    for r in reader:
                        batch.append(
                            (
                                to_date(r.get("ranking_date")),
                                to_int(r.get("rank")),
                                to_int(r.get("player")),
                                to_int(r.get("points")),
                            )
                        )
                        if len(batch) >= 5000:
                            cur.executemany(sql, batch)
                            count += len(batch)
                            batch = []
                    if batch:
                        cur.executemany(sql, batch)
                        count += len(batch)
                total += count
                print(f"Imported {path.name}: {count} rows")
            print(f"Rankings inserted (attempted): {total}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM atp_rankings")
            print("DB rankings:", cur.fetchone()[0])
            cur.execute("SELECT MIN(ranking_date), MAX(ranking_date), COUNT(DISTINCT ranking_date) FROM atp_rankings")
            print("Ranking date range and distinct weeks:", cur.fetchone())
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
