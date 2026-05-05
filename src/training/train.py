import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import INDEX_TO_LABEL, LABEL_TO_INDEX, VehicleOrientationDataset  # noqa: E402
from model import SUPPORTED_MODELS, create_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a v0.1 vehicle orientation classifier."
    )
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("generated_dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_random_seeds(seed):
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_loader(dataset_dir, split_csv, batch_size, num_workers, shuffle):
    dataset = VehicleOrientationDataset(
        dataset_dir=dataset_dir,
        split_csv=split_csv,
        transform=build_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy


def save_checkpoint(path, model, args, epoch, val_accuracy):
    torch.save(
        {
            "model_name": args.model_name,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "model_state_dict": model.state_dict(),
            "label_to_index": LABEL_TO_INDEX,
        },
        path,
    )


def write_metrics(path, metrics):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("training_runs") / f"{args.model_name}_run_001"

    print(f"[Seed] Using random seed: {args.seed}")
    set_random_seeds(args.seed)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        args.dataset_dir,
        args.splits_dir / "train.csv",
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        args.dataset_dir,
        args.splits_dir / "val.csv",
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    model = create_model(
        model_name=args.model_name,
        num_classes=len(LABEL_TO_INDEX),
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "model_name": args.model_name,
        "dataset_dir": str(args.dataset_dir),
        "splits_dir": str(args.splits_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "device": str(device),
        "seed": args.seed,
        "pretrained": not args.no_pretrained,
    }
    save_json(args.output_dir / "training_config.json", config)
    save_json(
        args.output_dir / "label_mapping.json",
        {
            "label_to_index": LABEL_TO_INDEX,
            "index_to_label": {str(key): value for key, value in INDEX_TO_LABEL.items()},
        },
    )

    best_val_accuracy = -1.0
    metrics = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        metrics.append(row)
        write_metrics(args.output_dir / "metrics.csv", metrics)

        save_checkpoint(
            args.output_dir / "last_model.pt",
            model,
            args,
            epoch,
            val_accuracy,
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model,
                args,
                epoch,
                val_accuracy,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    print()
    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
