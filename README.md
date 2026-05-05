# Vehicle Orientation Synthetic Dataset Generator

This project generates a clean synthetic image dataset in Blender for vehicle orientation classification from a single image.

The current scope includes deterministic dataset generation, dataset inspection, and a v0.1 sanity-check PyTorch training pipeline. It does not implement domain randomization, randomized backgrounds, randomized lighting, randomized cameras, texture randomization, blur, noise, or complex scene variation.

## Project Structure

```text
vehicle_orientation_project/
|-- assets/
|   `-- models/
|-- configs/
|   `-- generation_config.json
|-- generated_dataset/
|   |-- approaching/
|   |-- leaving/
|   |-- moving_left/
|   |-- moving_right/
|   `-- metadata.csv
|-- src/
|   `-- generation/
|       |-- generate_dataset.py
|       |-- blender_scene.py
|       |-- model_utils.py
|       |-- label_utils.py
|       `-- metadata_writer.py
`-- README.md
```

## Model Placement

Place vehicle models in:

```text
assets/models/
```

Supported formats are:

- `.glb`
- `.gltf`
- `.fbx`
- `.obj`

Models should be pre-oriented so the front of the vehicle points along Blender `+Y` at yaw `0` degrees.

## Running the Generator

Run the script with Blender in background mode:
You can generate a test set to see each model is loaded correctly:
```bash 
blender --background --python .\src\generation\generate_dataset.py -- --models-dir .\assets\models --output-dir .\generated_dataset_test --angle-step 90 --image-size 256 --render-samples 16
```
Or to generate the full dataset
```bash
blender --background --python src/generation/generate_dataset.py -- --models-dir assets/models --output-dir generated_dataset --angle-step 10 --image-size 512 --render-samples 64
```

You can also provide a JSON config:

```bash
blender --background --python src/generation/generate_dataset.py -- --config configs/generation_config.json --models-dir assets/models --output-dir generated_dataset
```

Explicit command-line values override config values for `angle-step`, `image-size`, and `render-samples`.

Relative paths are resolved from the project root, so `--output-dir generated_dataset` writes into this project's `generated_dataset/` folder even if Blender is started from another working directory.

## Orientation Labels

The script rotates each model around Blender's vertical Z axis. This is yaw rotation.

Yaw angles are mapped to labels as follows:

- `315 <= yaw < 360` or `0 <= yaw < 45`: `leaving`
- `45 <= yaw < 135`: `moving_right`
- `135 <= yaw < 225`: `approaching`
- `225 <= yaw < 315`: `moving_left`

The generated dataset is intended for a later CNN classifier that predicts vehicle orientation from a single rendered image.

## Deterministic Scene

Every render uses the same simple scene:

- Camera lens: `50 mm`
- Front/rear auto-framing: height `1.15`, target z `0.8`, margin `1.10`
- Side auto-framing: height `1.15`, target z `0.8`, margin `1.15`
- One area light at `(0, -4, 6)`
- Plain white background
- Square PNG output

The generator uses deterministic auto-framing based on the rotated model's projected bounding box. Front and rear renders use a tighter margin so the vehicle appears larger. Side renders use a slightly wider margin to reduce horizontal clipping. No random camera placement or automatic cropping is used.

The only changing variables are the 3D model and the yaw angle.

## Dataset Inspection

After generating a dataset, inspect it before using it for training:

```bash
python src/dataset_tools/inspect_dataset.py --dataset-dir generated_dataset
```

The inspector checks metadata/image consistency, class folders, label counts,
model counts, missing images, orphan images, invalid labels, unsafe paths, and
simple class imbalance. It exits with code `0` for valid datasets or warnings
only, and code `1` for critical validation issues.

## Training Pipeline v0.1

This is a sanity-check training pipeline, not final research-quality benchmarking.
With only a few vehicle models, results are not final. Later, when more models
are available, use a model-level train/validation/test split instead of the
current image-level split.

Install the training dependencies:

```bash
pip install -r requirements.txt
```

Create one deterministic split that both models will share:

```bash
python src/training/split_dataset.py --dataset-dir generated_dataset --output-dir splits --seed 42
```

Train and evaluate ResNet18:

```bash
python src/training/train.py --model-name resnet18 --dataset-dir generated_dataset --splits-dir splits --output-dir training_runs/resnet18_run_001 --epochs 10 --batch-size 16
python src/training/evaluate.py --model-name resnet18 --dataset-dir generated_dataset --splits-dir splits --checkpoint training_runs/resnet18_run_001/best_model.pt
```

Train and evaluate MobileNetV3 Small:

```bash
python src/training/train.py --model-name mobilenet_v3_small --dataset-dir generated_dataset --splits-dir splits --output-dir training_runs/mobilenet_v3_small_run_001 --epochs 10 --batch-size 16
python src/training/evaluate.py --model-name mobilenet_v3_small --dataset-dir generated_dataset --splits-dir splits --checkpoint training_runs/mobilenet_v3_small_run_001/best_model.pt
```

Compare the two runs:

```bash
python src/training/compare_models.py --runs training_runs/resnet18_run_001 training_runs/mobilenet_v3_small_run_001 --output training_runs/model_comparison.csv
```

Both models use the same split, preprocessing, optimizer, learning rate, batch
size, and epoch count by default so the comparison is fair for this v0.1 check.
