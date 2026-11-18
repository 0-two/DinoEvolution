"""Assemble the README animation from frames captured by ``web/scripts/capture-frames.mjs``.

    uv run python -m sim.figures.gif

Reads ``report/media/frames/*.png`` and writes ``report/media/live-evolution.gif``, then deletes
the frames. The page is mostly flat colour, so a small palette and frame differencing keep the
file well under the 8 MB limit; the script fails if it is over.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

MEDIA = Path(__file__).resolve().parents[2] / "report" / "media"
FRAMES = MEDIA / "frames"
OUT = MEDIA / "live-evolution.gif"
LIMIT_BYTES = 8 * 1024 * 1024
FRAME_MS = 130
COLOURS = 40


def main() -> int:
    paths = sorted(FRAMES.glob("frame_*.png"))
    if not paths:
        print(f"no frames in {FRAMES}; run web/scripts/capture-frames.mjs first")
        return 1
    frames = [Image.open(p).convert("RGB") for p in paths]
    # One shared palette from a few frames across the run, so colours do not flicker.
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for i, idx in enumerate((0, len(frames) // 2, len(frames) - 1)):
        sample.paste(frames[idx], (0, i * frames[0].height))
    palette = sample.quantize(colors=COLOURS, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(
        OUT,
        save_all=True,
        append_images=quantised[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({len(frames)} frames, {size / 1024 / 1024:.2f} MB)")
    shutil.rmtree(FRAMES)
    if size > LIMIT_BYTES:
        print("the GIF is over 8 MB; use fewer frames or a smaller crop")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
