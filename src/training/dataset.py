import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_INDEX = {
    "approaching": 0,
    "leaving": 1,
    "moving_left": 2,
    "moving_right": 3,
}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class VehicleOrientationDataset(Dataset):
    """PyTorch Dataset backed by a generated split CSV."""

    def __init__(self, dataset_dir, split_csv, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.split_csv = Path(split_csv)
        self.transform = transform
        self.rows = self._read_rows(self.split_csv)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.dataset_dir / row["relative_path"]
        label_index = LABEL_TO_INDEX[row["label"]]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, label_index

    @staticmethod
    def _read_rows(split_csv):
        with Path(split_csv).open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        for row in rows:
            label = row.get("label", "")
            if label not in LABEL_TO_INDEX:
                raise ValueError(f"Unsupported label in split CSV: {label}")
            if not row.get("relative_path"):
                raise ValueError("Split CSV contains a row without relative_path.")

        return rows
