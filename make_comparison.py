#!/usr/bin/env python3
"""
Layout per scene:
  [ Director3D | SplatFlow  ]   |  Ours (VIST3A)
  [ Prometheus | VideoRFSpl ]   |  (tall, prominent)
Prompt text header at top.
"""

import os
import subprocess
import sys
import textwrap

BASE = "asset/qualitative_comparison"
OUTPUT = "asset/comparison_all.mp4"
TMP_DIR = "/tmp/vist3a_rows"

# Baseline grid: 2x2, each cell this size
CELL_W = 256
CELL_H = 256
LABEL_H = 34  # space below each video for method name
FONT_SIZE_LABEL = 14
FONT_SIZE_PROMPT = 13

PROMPT_LINE_H = 22
PROMPT_LINES = 3
PROMPT_H = PROMPT_LINES * PROMPT_LINE_H + 14
WRAP_CHARS = 110

# Ours panel: same total height as 2x2 grid
GRID_W = CELL_W * 2  # 512
GRID_H = (CELL_H + LABEL_H) * 2  # 580
OURS_VID_H = GRID_H - LABEL_H  # video portion height
OURS_VID_W = CELL_W * 2  # 512 — square-ish

BASELINES = ["director3d", "splatflow", "prometheus3d", "videorfsplat"]
BASE_LABELS = ["Director3D", "SplatFlow", "Prometheus3D", "VideoRFSplat"]

SCENES = [
    {
        "name": "chinese",
        "ours": "ours",
        "prompt": (
            "An Asian restaurant, possibly Chinese, depicted in a street view scene. "
            "The entrance is marked by a large blue sign with Chinese characters. "
            "In front there is a prominent gray awning. Trees and bushes add greenery."
        ),
    },
    {
        "name": "baby",
        "ours": "gs",
        "prompt": (
            "A small infant with silver-framed glasses sits on a plush white bed "
            "in a pale yellow onesie, holding a colorful picture book. "
            "Plush toys including a fluffy blue bear and soft green frog surround the infant."
        ),
    },
    {
        "name": "blue_bird",
        "ours": "ours",
        "prompt": "A bluebird perched on a tree branch.",
    },
    {
        "name": "chips_castle",
        "ours": "ours",
        "prompt": (
            "A castle built from golden tortilla chips stands amidst a flowing river of red salsa. "
            "Tiny animated burritos meander along the banks. "
            "The whimsical landscape is set upon a large plate."
        ),
    },
]

os.makedirs(TMP_DIR, exist_ok=True)


def esc(t):
    return (
        t.replace("\\", "\\\\")
        .replace("'", "\u2019")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def build_row(scene, row_out):
    # inputs: 0-3 baselines, 4 = ours
    inputs = []
    for m in BASELINES:
        inputs += ["-i", f"{BASE}/{scene['name']}/{m}.mp4"]
    inputs += ["-i", f"{BASE}/{scene['name']}/{scene['ours']}.mp4"]

    lines = textwrap.wrap(scene["prompt"], width=WRAP_CHARS)[:PROMPT_LINES]
    parts = []

    # ── Baselines: scale → letterbox → pad label → draw label ──────────────
    for i, label in enumerate(BASE_LABELS):
        parts.append(
            f"[{i}:v]"
            f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_W}:{CELL_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"pad=iw:ih+{LABEL_H}:0:0:color=#1a1a1a,"
            f"drawtext=text='{esc(label)}':"
            f"fontcolor=white:fontsize={FONT_SIZE_LABEL}:"
            f"x=(w-text_w)/2:y=h-{LABEL_H - 6}"
            f"[b{i}]"
        )

    # ── 2x2 grid ────────────────────────────────────────────────────────────
    parts.append("[b0][b1]hstack=inputs=2[top]")
    parts.append("[b2][b3]hstack=inputs=2[bot]")
    parts.append("[top][bot]vstack=inputs=2[grid]")

    # ── Ours: scale → letterbox → pad label → draw label → border ───────────
    parts.append(
        f"[4:v]"
        f"scale={OURS_VID_W}:{OURS_VID_H}:force_original_aspect_ratio=decrease,"
        f"pad={OURS_VID_W}:{OURS_VID_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"pad=iw:ih+{LABEL_H}:0:0:color=#1a1a1a,"
        f"drawtext=text='Ours (VIST3A)':"
        f"fontcolor=yellow:fontsize={FONT_SIZE_LABEL + 2}:"
        f"x=(w-text_w)/2:y=h-{LABEL_H - 6},"
        f"drawbox=x=2:y=2:w=iw-4:h=ih-4:color=0x3273dc@1.0:t=5"
        f"[ours]"
    )

    # ── Side by side: grid | ours ────────────────────────────────────────────
    parts.append("[grid][ours]hstack=inputs=2[combined]")

    # ── Pad top for prompt ───────────────────────────────────────────────────
    parts.append(f"[combined]pad=iw:ih+{PROMPT_H}:0:{PROMPT_H}:color=#111111[padded]")

    # ── Draw prompt lines ────────────────────────────────────────────────────
    prev = "padded"
    for li, line in enumerate(lines):
        y = 8 + li * PROMPT_LINE_H
        nxt = f"pl{li}"
        parts.append(
            f"[{prev}]drawtext=text='{esc(line)}':"
            f"fontcolor=white:fontsize={FONT_SIZE_PROMPT}:"
            f"x=12:y={y}[{nxt}]"
        )
        prev = nxt

    # Rename last label → [out]
    parts[-1] = parts[-1][: parts[-1].rfind("[")] + "[out]"

    fc = ";".join(parts)
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            fc,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            row_out,
        ]
    )

    print(f"\n▶  Building: {scene['name']}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        sys.exit(1)
    print(f"   ✓ {row_out}")


def concat_rows(row_files, output):
    lst = f"{TMP_DIR}/concat.txt"
    with open(lst, "w") as f:
        for r in row_files:
            f.write(f"file '{os.path.abspath(r)}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        lst,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output,
    ]
    print(f"\n▶  Concatenating → {output}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        sys.exit(1)
    print(f"\n🎉  Done! → {output}")


if __name__ == "__main__":
    rows = []
    for scene in SCENES:
        out = f"{TMP_DIR}/row_{scene['name']}.mp4"
        build_row(scene, out)
        rows.append(out)
    concat_rows(rows, OUTPUT)
