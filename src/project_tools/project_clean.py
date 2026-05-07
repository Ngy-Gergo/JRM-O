from pathlib import Path
import shutil


QUESTIONS = [
    (
        "Do you want to clean the generated dataset folder? [Y] yes [N] no",
        ["generated_dataset"],
    ),
    (
        "Do you want to clean the generated test dataset folder? [Y] yes [N] no",
        ["generated_dataset_test"],
    ),
    (
        "Do you want to clean the splits folder? [Y] yes [N] no",
        ["splits"],
    ),
    (
        "Do you want to clean the test splits folder? [Y] yes [N] no",
        ["splits_test"],
    ),
    (
        "Do you want to clean the training runs folder? [Y] yes [N] no",
        ["training_runs"],
    ),
    (
        "Do you want to clean quick training runs only? [Y] yes [N] no",
        [
            "training_runs/quick_resnet18",
            "training_runs/quick_mobilenet_v3_small",
            "training_runs/quick_model_comparison.csv",
            "training_runs/quick_model_comparison.json",
        ],
    ),
    (
        "Do you want to clean model comparison files? [Y] yes [N] no",
        [
            "training_runs/model_comparison.csv",
            "training_runs/model_comparison.json",
        ],
    ),
]

KNOWN_TARGETS = {
    target
    for _, targets in QUESTIONS
    for target in targets
}

PROTECTED_TARGETS = {
    "src",
    "assets",
    "assets/models",
    "configs",
    "README.md",
    "requirements.txt",
    ".git",
}


def project_root():
    return Path(__file__).resolve().parents[2]


def ask_yes_no(question):
    while True:
        answer = input(f"{question} ").strip()

        if answer in {"Y", "y", "yes"}:
            return True
        if answer in {"N", "n", "no"}:
            return False

        print("Please answer Y or N.")


def delete_path(path):
    root = project_root()
    resolved_path = path.resolve()

    print(f"Target: {resolved_path}")

    try:
        relative_path = resolved_path.relative_to(root).as_posix()
    except ValueError:
        print(f"Skipped target outside project root: {resolved_path}")
        return "skipped"

    if relative_path not in KNOWN_TARGETS or relative_path in PROTECTED_TARGETS:
        print(f"Skipped protected or unknown target: {resolved_path}")
        return "skipped"

    if not resolved_path.exists():
        print(f"Not found: {resolved_path}")
        return "missing"

    if resolved_path.is_dir():
        shutil.rmtree(resolved_path)
    else:
        resolved_path.unlink()

    print(f"Deleted: {resolved_path}")
    return "deleted"


def print_summary(deleted_targets, skipped_targets, missing_targets):
    print()
    print("Cleanup summary")
    print("Deleted targets:")
    if deleted_targets:
        for target in deleted_targets:
            print(f"- {target}")
    else:
        print("- none")

    print("Skipped targets:")
    if skipped_targets:
        for target in skipped_targets:
            print(f"- {target}")
    else:
        print("- none")

    print("Missing targets:")
    if missing_targets:
        for target in missing_targets:
            print(f"- {target}")
    else:
        print("- none")


def main():
    root = project_root()
    deleted_targets = []
    skipped_targets = []
    missing_targets = []

    print("Cleaning process started...")

    for question, targets in QUESTIONS:
        if ask_yes_no(question):
            for target in targets:
                path = root / target
                status = delete_path(path)

                if status == "deleted":
                    deleted_targets.append(target)
                elif status == "missing":
                    missing_targets.append(target)
                else:
                    skipped_targets.append(target)
        else:
            skipped_targets.extend(targets)

    print_summary(deleted_targets, skipped_targets, missing_targets)


if __name__ == "__main__":
    main()
