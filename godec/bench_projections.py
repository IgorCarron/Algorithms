import csv
import os
import time

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import array, column_stack

from godec import godec


PROJECTIONS = [
    "gaussian",
    "rademacher",
    "achlioptas",
    "sparse_jl",
    "srht",
    "srft",
    "block_sparse",
]

FILES = [
    ("demo.avi", "dataset/demo.avi"),
    ("highway.mpg", "dataset/highway.mpg"),
    ("forbiggerblazes.mp4", "dataset/forbiggerblazes.mp4"),
]

FRAMES = 30


def load_frames(path, max_frames):
    cap = cv.VideoCapture(path)
    M = None
    i = 0
    while cap.isOpened() and i < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        F = frame.T.reshape(-1)
        if i == 0:
            M = array([F]).T
        else:
            M = column_stack((M, F))
        i += 1
    cap.release()
    if M is None:
        raise RuntimeError(f"Failed to read frames from {path}")
    return M


def main():
    tag = os.environ.get("BENCH_TAG", "current")
    csv_path = f"bench_projections_pixels_{tag}.csv"
    png_path = f"bench_projections_pixels_{tag}.png"
    results = {}
    for label, path in FILES:
        M = load_frames(path, FRAMES).astype("float64", copy=False)
        total_pixels = M.shape[0] * M.shape[1]
        results[label] = {"pixels": total_pixels, "times": {}}

        for proj in PROJECTIONS:
            t0 = time.perf_counter()
            godec(M, projection=proj)
            elapsed = time.perf_counter() - t0
            results[label]["times"][proj] = elapsed
            print(label, proj, M.shape, elapsed)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "pixels", "projection", "seconds"])
        for label, data in results.items():
            for proj, secs in data["times"].items():
                writer.writerow([label, data["pixels"], proj, secs])

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2f4858", "#33658a", "#86bbd8", "#f6ae2d", "#f26419", "#5b8e7d"]

    for proj, color in zip(PROJECTIONS, colors):
        xs = []
        ys = []
        labels = []
        for label, _ in FILES:
            xs.append(results[label]["pixels"])
            ys.append(results[label]["times"][proj])
            labels.append(label)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        labels = [labels[i] for i in order]

        ax.plot(xs, ys, marker="o", label=proj, color=color)
        for x, y, lbl in zip(xs, ys, labels):
            ax.text(x, y, lbl, ha="center", va="bottom", fontsize=7, color=color)

    ax.set_xlabel("Total pixels (frames * width * height)")
    ax.set_ylabel("Seconds (30 frames)")
    ax.set_title("GoDec (Rust) - 30-frame time by projection")
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
