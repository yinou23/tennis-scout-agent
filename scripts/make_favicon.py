"""Draw the TSA favicon as a multi-size .ico.

Design: a bold green "%" (the site is about odds/probabilities),
same shapes as web/public/favicon.svg. Run from the repo root:

    python3 scripts/make_favicon.py

Writes web/public/favicon.ico (16/32/48px).
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
GREEN = (34, 197, 94, 255)  # #22c55e


def main():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = SIZE / 100.0
    # Top-left dot and bottom-right dot.
    d.ellipse([15 * s, 13 * s, 45 * s, 43 * s], fill=GREEN)
    d.ellipse([55 * s, 57 * s, 85 * s, 87 * s], fill=GREEN)
    # Diagonal slash, upper-right to lower-left.
    w = max(2, int(13 * s))
    d.line([(74 * s, 20 * s), (26 * s, 80 * s)], fill=GREEN, width=w)
    out = ROOT / "web" / "public" / "favicon.ico"
    img.save(out, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
