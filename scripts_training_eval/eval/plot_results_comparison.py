import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


LOWER_IS_BETTER = {
    "chamfer_test": True,
    "track_test": True,
    "psnr_test": False,
    "ssim_test": False,
    "lpips_test": True,
    "iou_test": False,
}


def discover_methods(results_dir):
    methods = set()
    for filename in os.listdir(results_dir):
        if filename.endswith("_chamfer.csv"):
            methods.add(filename[: -len("_chamfer.csv")])
        elif filename.endswith("_track.csv"):
            methods.add(filename[: -len("_track.csv")])
        elif filename.endswith("_render.txt"):
            methods.add(filename[: -len("_render.txt")])
    return sorted(methods)


def read_chamfer_csv(path):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Case Name"]] = {
                "train": float(row["Train Chamfer Error"]),
                "test": float(row["Test Chamfer Error"]),
            }
    return rows


def read_track_csv(path):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Case Name"]] = {
                "train": float(row["Train Track Error"]),
                "test": float(row["Test Track Error"]),
            }
    return rows


def read_render_txt(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    overall = {}
    scene_metrics = {}
    overall_pattern = re.compile(r"Overall (\w+) \((train|test)\):\s+([-+eE0-9.]+)")
    scene_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s+\|\s+(.+)$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = overall_pattern.search(line)
            if match:
                metric = match.group(1).lower()
                split = match.group(2).lower()
                overall[f"{metric}_{split}"] = float(match.group(3))
                continue

            if "|" not in line or line.strip().startswith("Scene") or set(line.strip()) == {"-"}:
                continue

            match = scene_pattern.match(line.rstrip())
            if not match:
                continue

            scene = match.group(1)
            if scene == "OVERALL":
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 9:
                continue

            try:
                scene_metrics[scene] = {
                    "psnr_train": float(parts[1]),
                    "ssim_train": float(parts[2]),
                    "lpips_train": float(parts[3]),
                    "iou_train": float(parts[4]),
                    "psnr_test": float(parts[5]),
                    "ssim_test": float(parts[6]),
                    "lpips_test": float(parts[7]),
                    "iou_test": float(parts[8]),
                }
            except ValueError:
                continue

    if not overall and not scene_metrics:
        return None

    return {"overall": overall, "scene": scene_metrics}


def load_method_results(results_dir, methods):
    data = {}
    for method in methods:
        data[method] = {
            "chamfer": read_chamfer_csv(os.path.join(results_dir, f"{method}_chamfer.csv")),
            "track": read_track_csv(os.path.join(results_dir, f"{method}_track.csv")),
            "render": read_render_txt(os.path.join(results_dir, f"{method}_render.txt")),
        }
    return data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_overall_bar_chart(methods, values_by_method, metric_name, out_path):
    available = [(m, v) for m, v in values_by_method if v is not None and not math.isnan(v)]
    if not available:
        return

    labels = [m for m, _ in available]
    values = [v for _, v in available]
    colors = ["#4C78A8" if m == "baseline" else "#F58518" for m in labels]

    plt.figure(figsize=(max(6, len(labels) * 1.2), 4.8))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel(metric_name)
    plt.title(f"Overall {metric_name} Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_case_delta_plot(baseline_values, method_values, metric_name, out_path, lower_is_better):
    shared_cases = sorted(set(baseline_values) & set(method_values))
    if not shared_cases:
        return

    deltas = []
    for case in shared_cases:
        base = baseline_values[case]
        cur = method_values[case]
        if lower_is_better:
            delta = base - cur
        else:
            delta = cur - base
        deltas.append(delta)

    plt.figure(figsize=(max(10, len(shared_cases) * 0.45), 5.2))
    colors = ["#54A24B" if d >= 0 else "#E45756" for d in deltas]
    plt.bar(shared_cases, deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Improvement vs baseline")
    plt.title(f"{metric_name}: improvement per case")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=65, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def extract_case_metric(results, group, split):
    if not results:
        return None
    return {case: metrics[split] for case, metrics in results.items()}


def extract_render_scene_metric(render_results, key):
    if not render_results or "scene" not in render_results:
        return None
    return {case: metrics[key] for case, metrics in render_results["scene"].items()}


def write_summary(results_dir, out_dir, methods, all_results):
    methods_tag = "_".join(m for m in methods if m != "baseline")
    summary_path = os.path.join(out_dir, f"{methods_tag}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Methods discovered:\n")
        for method in methods:
            f.write(f"- {method}\n")
        f.write("\nOverall test metrics:\n")

        metric_extractors = {
            "chamfer_test": lambda r: (
                None if not r["chamfer"] else mean(v["test"] for v in r["chamfer"].values())
            ),
            "track_test": lambda r: (
                None if not r["track"] else mean(v["test"] for v in r["track"].values())
            ),
            "psnr_test": lambda r: None if not r["render"] else r["render"]["overall"].get("psnr_test"),
            "ssim_test": lambda r: None if not r["render"] else r["render"]["overall"].get("ssim_test"),
            "lpips_test": lambda r: None if not r["render"] else r["render"]["overall"].get("lpips_test"),
            "iou_test": lambda r: None if not r["render"] else r["render"]["overall"].get("iou_test"),
        }

        for metric_name, extractor in metric_extractors.items():
            f.write(f"\n{metric_name}:\n")
            scored = []
            for method in methods:
                value = extractor(all_results[method])
                if value is not None:
                    scored.append((method, value))

            if not scored:
                f.write("  no data\n")
                continue

            reverse = not LOWER_IS_BETTER[metric_name]
            scored.sort(key=lambda x: x[1], reverse=reverse)
            for method, value in scored:
                f.write(f"  {method}: {value:.6f}\n")


def mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description="Plot baseline-vs-method comparison figures from results/")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/plots")
    parser.add_argument(
        "--methods",
        type=str,
        required=True,
        help="Comma-separated method tags. Example: baseline,output_2,output_3",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if not methods:
        raise SystemExit("At least one method must be provided via --methods")
    if "baseline" not in methods:
        raise SystemExit("baseline_* results are required for comparison plots")

    ensure_dir(args.output_dir)
    all_results = load_method_results(args.results_dir, methods)

    overall_specs = {
        "chamfer_test": lambda r: None if not r["chamfer"] else mean(v["test"] for v in r["chamfer"].values()),
        "track_test": lambda r: None if not r["track"] else mean(v["test"] for v in r["track"].values()),
        "psnr_test": lambda r: None if not r["render"] else r["render"]["overall"].get("psnr_test"),
        "ssim_test": lambda r: None if not r["render"] else r["render"]["overall"].get("ssim_test"),
        "lpips_test": lambda r: None if not r["render"] else r["render"]["overall"].get("lpips_test"),
        "iou_test": lambda r: None if not r["render"] else r["render"]["overall"].get("iou_test"),
    }

    methods_tag = "_".join(m for m in methods if m != "baseline")
    for metric_name, extractor in overall_specs.items():
        values = [(method, extractor(all_results[method])) for method in methods]
        save_overall_bar_chart(
            methods,
            values,
            metric_name,
            os.path.join(args.output_dir, f"{methods_tag}_overall_{metric_name}.png"),
        )

    baseline = all_results["baseline"]
    case_specs = {
        "chamfer_test": (
            extract_case_metric(baseline["chamfer"], "chamfer", "test"),
            lambda method_result: extract_case_metric(method_result["chamfer"], "chamfer", "test"),
        ),
        "track_test": (
            extract_case_metric(baseline["track"], "track", "test"),
            lambda method_result: extract_case_metric(method_result["track"], "track", "test"),
        ),
        "psnr_test": (
            extract_render_scene_metric(baseline["render"], "psnr_test"),
            lambda method_result: extract_render_scene_metric(method_result["render"], "psnr_test"),
        ),
        "ssim_test": (
            extract_render_scene_metric(baseline["render"], "ssim_test"),
            lambda method_result: extract_render_scene_metric(method_result["render"], "ssim_test"),
        ),
        "lpips_test": (
            extract_render_scene_metric(baseline["render"], "lpips_test"),
            lambda method_result: extract_render_scene_metric(method_result["render"], "lpips_test"),
        ),
        "iou_test": (
            extract_render_scene_metric(baseline["render"], "iou_test"),
            lambda method_result: extract_render_scene_metric(method_result["render"], "iou_test"),
        ),
    }

    for method in methods:
        if method == "baseline":
            continue
        for metric_name, (baseline_values, extractor) in case_specs.items():
            method_values = extractor(all_results[method])
            if baseline_values is None or method_values is None:
                continue
            save_case_delta_plot(
                baseline_values,
                method_values,
                f"{method} vs baseline ({metric_name})",
                os.path.join(args.output_dir, f"{method}_vs_baseline_{metric_name}.png"),
                lower_is_better=LOWER_IS_BETTER[metric_name],
            )

    write_summary(args.results_dir, args.output_dir, methods, all_results)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
