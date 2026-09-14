"""
Run the detector on a still or a video.

Examples:
  python run.py media/PennAir\\ 2024\\ App\\ Static.png
  python run.py media/PennAir\\ 2024\\ App\\ Dynamic.mp4 -o results/dynamic.mp4
  python run.py media/PennAir\\ 2024\\ App\\ Dynamic\\ Hard.mp4 -o results/hard.mp4
"""

import argparse
import os
import sys
import time

import cv2

from detect import draw, find_shapes


def process_image(path, out_path, show=True):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"couldn't read {path}")
    shapes, _mask = find_shapes(img)
    vis = draw(img, shapes)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"wrote {out_path}  ({len(shapes)} shapes)")
    for s in shapes:
        print(f"  {s['name']:10s} center=({s['center'][0]:.0f}, {s['center'][1]:.0f})")
    if show:
        cv2.imshow("shapes", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macos needs one more event-loop tick to close


def process_video(path, out_path, show=True, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"couldn't open {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{w}x{h} @ {fps:.1f} fps, {n} frames")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # mp4v is the one that actually writes on my mac; h264 via opencv
    # gave me a file nothing could play.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    i = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shapes, _ = find_shapes(frame)
        vis = draw(frame, shapes)
        writer.write(vis)
        i += 1
        if i % 30 == 0:
            dt = time.time() - t0
            print(f"  frame {i}/{n}  {i/dt:.1f} fps")
        if show:
            cv2.imshow("shapes", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and i >= max_frames:
            break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    dt = time.time() - t0
    print(f"wrote {out_path}  ({i} frames, {i/dt:.1f} fps processing)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    default_out = os.path.join("results", "out" + (ext if ext else ".png"))
    out = args.output or default_out
    show = not args.no_show

    if ext in {".png", ".jpg", ".jpeg"}:
        process_image(args.input, out, show=show)
    else:
        process_video(args.input, out, show=show, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
