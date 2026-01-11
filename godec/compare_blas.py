import csv

import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pixels"] = int(row["pixels"])
            row["seconds"] = float(row["seconds"])
            rows.append(row)
    return rows


def index_rows(rows):
    data = {}
    for row in rows:
        proj = row["projection"]
        data.setdefault(proj, {})
        data[proj][row["pixels"]] = row["seconds"]
    return data


def best_by_video(rows):
    best = {}
    for row in rows:
        key = (row["video"], row["pixels"])
        cur = best.get(key)
        if cur is None or row["seconds"] < cur["seconds"]:
            best[key] = row
    return best


def main():
    blas_rows = load_csv("bench_projections_pixels_blas.csv")
    noblas_rows = load_csv("bench_projections_pixels_noblas.csv")

    blas = index_rows(blas_rows)
    noblas = index_rows(noblas_rows)

    projections = sorted(blas.keys())
    colors = [
        "#2f4858",
        "#33658a",
        "#86bbd8",
        "#f6ae2d",
        "#f26419",
        "#5b8e7d",
        "#9e2a2b",
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for proj, color in zip(projections, colors):
        xs = sorted(blas[proj].keys())
        y_blas = [blas[proj][x] for x in xs]
        y_noblas = [noblas[proj][x] for x in xs]
        ax.plot(xs, y_blas, marker="o", color=color, label=f"{proj} (BLAS)")
        ax.plot(xs, y_noblas, marker="x", linestyle="--", color=color, label=f"{proj} (no BLAS)")

    ax.set_xlabel("Total pixels (frames * width * height)")
    ax.set_ylabel("Seconds (30 frames)")
    ax.set_title("GoDec - BLAS vs non-BLAS by projection")
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig("bench_projections_blas_vs_noblas.png", dpi=150)
    print("Saved bench_projections_blas_vs_noblas.png")

    blas_best = best_by_video(blas_rows)
    noblas_best = best_by_video(noblas_rows)
    with open("bench_projections_best.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "pixels", "blas_projection", "blas_seconds", "noblas_projection", "noblas_seconds"])
        for key in sorted(blas_best.keys()):
            b = blas_best[key]
            nb = noblas_best[key]
            writer.writerow([
                b["video"],
                b["pixels"],
                b["projection"],
                b["seconds"],
                nb["projection"],
                nb["seconds"],
            ])
    print("Saved bench_projections_best.csv")


if __name__ == "__main__":
    main()
