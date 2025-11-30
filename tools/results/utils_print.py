from pathlib import Path

from src.data.chapters import Chapters
from tools.results.metrics import MetricsExperiment
from tools.results.utils import (
    find_common_path,
    find_matching_paths,
    format_number_with_z,
)

vidc_dir = "dataset" if Path("dataset/").exists() else "../../dataset/"

chp = Chapters(vidc_dir=vidc_dir, subset="eval")


def print_exp_metrics(exp_name, exp_dir, subset_test):
    metrics = MetricsExperiment(chp, exp_dir=exp_dir, strict=False)
    results_overlap = metrics.compute_metrics_overlap(
        subset_test, metrics=("F1", "Avg. TIoU")
    )
    results_soda = metrics.compute_metrics_soda(subset_test)
    results_captions = metrics.compute_metrics_captions(subset_test, metrics=("CIDEr",))

    # Collect all metrics in a dictionary
    all_metrics = {}
    if results_overlap:
        all_metrics.update(
            {
                # "P (overlap)": f"{results_overlap['P']:.1f}",
                # "R (overlap)": f"{results_overlap['R']:.1f}",
                "F1 (overlap)": format_number_with_z(results_overlap["F1"]),
                "Avg. TIoU": format_number_with_z(results_overlap["Avg. TIoU"]),
            }
        )
    if results_soda:
        all_metrics["F1 (SODA)"] = format_number_with_z(results_soda["F1"])
    if results_captions:
        all_metrics.update(
            {
                "CIDEr": format_number_with_z(results_captions["CIDEr"], num_digits=2),
                # "METEOR": f"{results_captions['METEOR']:.1f}",
            }
        )

    # Print metrics in table format
    if all_metrics:
        # print(exp_name, "&" + " & ".join(all_metrics.keys()) + " \\\\")
        metrics_str = " & ".join(all_metrics.values())
        print(f"{exp_name} & {metrics_str} \\\\")


def print_n_captions(exp_name, exp_dir, subset_test):
    metrics = MetricsExperiment(chp, exp_dir=exp_dir, strict=False)
    n_captions = metrics.get_n_captions(subset_test)
    n_captions = format_number_with_z(n_captions, num_digits=3)
    print(f"& {exp_name} & {n_captions} \\\\")


def print_f1(exp_name, exp_dir, subset_test):
    metrics = MetricsExperiment(chp, exp_dir=exp_dir, strict=False)
    results_overlap = metrics.compute_metrics_overlap(subset_test, metrics=("F1",))

    # Collect all metrics in a dictionary
    all_metrics = {}
    if results_overlap:
        all_metrics.update(
            {
                "F1": format_number_with_z(results_overlap["F1"]),
            }
        )

    # Print metrics in table format
    if all_metrics:
        # print(exp_name, "&" + " & ".join(all_metrics.keys()) + " \\\\")
        metrics_str = " & ".join(all_metrics.values())
        print(f"{exp_name} & {metrics_str} \\\\")


def print_metrics(
    config_var,
    config_fix,
    subset_test="sml300_val",
    print_output="metrics",
):
    if isinstance(print_output, str):
        print_output = [print_output]
    assert isinstance(print_output, list), "print_output must be a list"
    assert set(print_output) <= {"n_captions", "metrics", "f1"}, "Invalid print_output"

    matching_paths = find_matching_paths(config_var, config_fix)
    # Find the common path prefix
    common_prefix = find_common_path(sum(matching_paths.values(), []))

    # Assert there is only one experiment dir per matching paths
    for exp_name, exp_dirs in matching_paths.items():
        print(exp_name)
        # assert exp_dirs, f"No experiments found for {exp_name}"
        for exp_dir in exp_dirs:
            relative_path = exp_dir.relative_to(common_prefix)
            print(f"\t{relative_path}")
        assert len(exp_dirs) <= 1, f"Multiple experiments found for {exp_name}"
    print("=" * 100)

    # Print results with relative paths
    for exp_name, exp_dirs in matching_paths.items():
        assert len(exp_dirs) == 1, (
            f"Multiple experiments found for {exp_name}: {exp_dirs}"
        )
        exp_dir = exp_dirs[0]
        relative_path = exp_dir.relative_to(common_prefix)
        if "n_captions" in print_output:
            print_n_captions(exp_name, exp_dir, subset_test)
        if "metrics" in print_output:
            print_exp_metrics(exp_name, exp_dir, subset_test)
        if "f1" in print_output:
            print_f1(exp_name, exp_dir, subset_test)
