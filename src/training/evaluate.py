import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import INDEX_TO_LABEL, LABEL_TO_INDEX, VehicleOrientationDataset  # noqa: E402
from model import SUPPORTED_MODELS, create_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained vehicle orientation classifier."
    )
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("generated_dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name and checkpoint_model_name != args.model_name:
        raise ValueError(
            f"Checkpoint model_name is '{checkpoint_model_name}', "
            f"but --model-name is '{args.model_name}'."
        )

    model = create_model(
        model_name=args.model_name,
        num_classes=len(LABEL_TO_INDEX),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device):
    num_classes = len(LABEL_TO_INDEX)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for true_label, predicted_label in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                confusion[int(true_label), int(predicted_label)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class_accuracy = {}
    for index, label in INDEX_TO_LABEL.items():
        class_total = int(confusion[index].sum())
        class_correct = int(confusion[index, index])
        per_class_accuracy[label] = class_correct / class_total if class_total else 0.0

    return accuracy, per_class_accuracy, confusion


def save_evaluation(run_dir, args, accuracy, per_class_accuracy, confusion):
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = [INDEX_TO_LABEL[index] for index in range(len(INDEX_TO_LABEL))]

    evaluation = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "test_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "labels": labels,
        "confusion_matrix": confusion.tolist(),
    }
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as json_file:
        json.dump(evaluation, json_file, indent=2)

    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label"] + labels)
        for index, label in enumerate(labels):
            writer.writerow([label] + confusion[index].tolist())


def print_results(accuracy, per_class_accuracy, confusion):
    labels = [INDEX_TO_LABEL[index] for index in range(len(INDEX_TO_LABEL))]
    print(f"Test accuracy: {accuracy:.4f}")
    print()
    print("Per-class accuracy")
    for label in labels:
        print(f"{label}: {per_class_accuracy[label]:.4f}")
    print()
    print("Confusion matrix")
    print(",".join(["true_label"] + labels))
    for index, label in enumerate(labels):
        values = [str(value) for value in confusion[index].tolist()]
        print(",".join([label] + values))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dataset = VehicleOrientationDataset(
        dataset_dir=args.dataset_dir,
        split_csv=args.splits_dir / "test.csv",
        transform=build_transform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = load_model(args, device)
    accuracy, per_class_accuracy, confusion = evaluate(model, dataloader, device)
    run_dir = args.checkpoint.parent
    save_evaluation(run_dir, args, accuracy, per_class_accuracy, confusion)
    print_results(accuracy, per_class_accuracy, confusion)
    print()
    print(f"Evaluation written to: {run_dir}")


if __name__ == "__main__":
    main()
