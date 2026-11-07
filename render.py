"""Turn recorded greedy games into GIF/MP4 clips for the video.

    python render.py                 -> videos/<run>_<pct>.gif  and  videos/<run>_progression.mp4
    python render.py dqn bad_myopic  -> only those runs
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CELL, HEADER = 40, 70
BG, GRID, INK, INK2 = (252, 252, 251), (230, 229, 224), (11, 11, 11), (82, 81, 78)
HEAD, BODY, FOOD = (28, 92, 171), (57, 135, 229), (227, 73, 72)
ENDINGS = {"wall": "died: hit the wall", "self": "died: bit itself", "starved": "starved",
           "timeout": "ran out the clock", "win": "filled the board!",
           "invalid": "answered with a non-move token"}
MAX_FRAMES = 300  # keep looping agents from producing endless clips


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_BIG, F_SMALL = font(22), font(16)


def draw(frame, size, title, subtitle, score):
    W = size * CELL
    im = Image.new("RGB", (W, W + HEADER), BG)
    d = ImageDraw.Draw(im)
    d.text((10, 8), title, fill=INK, font=F_BIG)
    d.text((10, 40), f"{subtitle}   score {score}", fill=INK2, font=F_SMALL)
    for i in range(size + 1):
        d.line([(i * CELL, HEADER), (i * CELL, HEADER + W)], fill=GRID)
        d.line([(0, HEADER + i * CELL), (W, HEADER + i * CELL)], fill=GRID)
    fx, fy = frame["food"]
    d.ellipse([fx * CELL + 8, HEADER + fy * CELL + 8, fx * CELL + CELL - 8, HEADER + fy * CELL + CELL - 8], fill=FOOD)
    for k, (x, y) in enumerate(frame["body"]):
        pad = 3 if k == 0 else 5
        d.rounded_rectangle([x * CELL + pad, HEADER + y * CELL + pad, x * CELL + CELL - pad,
                             HEADER + y * CELL + CELL - pad], radius=6, fill=HEAD if k == 0 else BODY)
    return im


def clip(name, replay_file):
    rp = json.loads(replay_file.read_text())
    pct = int(replay_file.stem)
    frames, score, out = rp["frames"][:MAX_FRAMES], 0, []
    for i, f in enumerate(frames):
        score = len(f["body"]) - 3
        out.append(draw(f, rp["size"], name, f"{pct}% of training", score))
    end = ENDINGS.get(rp["cause"], "cut")
    if len(rp["frames"]) > MAX_FRAMES:
        end = f"still going after {len(rp['frames'])} steps"
    last = out[-1].copy()
    ImageDraw.Draw(last).text((10 + 230, 40), end, fill=FOOD, font=F_SMALL)
    return out + [last] * 15


def render(name, runs="runs", out="videos", fps=12):
    Path(out).mkdir(exist_ok=True)
    all_frames = []
    for rf in sorted((Path(runs) / name / "replays").glob("*.json")):
        frames = clip(name, rf)
        frames[0].save(Path(out) / f"{name}_{rf.stem}.gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        all_frames += frames
    if shutil.which("ffmpeg") and all_frames:
        tmp = Path(out) / f".{name}_frames"
        tmp.mkdir(exist_ok=True)
        for i, f in enumerate(all_frames):
            f.save(tmp / f"{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        str(Path(out) / f"{name}_progression.mp4")], check=True)
        shutil.rmtree(tmp)
    print(f"{name}: {len(all_frames)} frames")


if __name__ == "__main__":
    args = sys.argv[1:]
    runs = args.pop(args.index("--runs") + 1) if "--runs" in args else "runs"
    out = args.pop(args.index("--out") + 1) if "--out" in args else "videos"
    args = [a for a in args if a not in ("--runs", "--out")]
    names = args or sorted(p.name for p in Path(runs).iterdir() if (p / "replays").exists())
    for n in names:
        render(n, runs, out)
