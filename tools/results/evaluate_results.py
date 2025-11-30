from pathlib import Path

from tools.results.utils_print import print_exp_metrics


def evaluate_results(exp_dir: Path, subset_test: str):
    print_exp_metrics(subset_test, exp_dir, subset_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_experiment_test_dir", type=Path)
    parser.add_argument("--subset", type=str, default="sml300_val")
    args = parser.parse_args()

    assert args.path_to_experiment_test_dir.exists(), (
        f"Path {args.path_to_experiment_test_dir} does not exist"
    )
    evaluate_results(args.path_to_experiment_test_dir, args.subset)
