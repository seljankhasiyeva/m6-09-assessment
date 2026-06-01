![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Final Assessment | Cat Detection v2 — Improve, Export to ONNX, Containerise & Compete

## Overview

This is the **final assessment** of Unit 6, and it has three parts that fit together:

1. **Improve** the YOLO26 cat detector you trained in the Week-1 assessment using techniques from Week 2 (larger backbones, stronger augmentation, transfer learning, hyperparameter tuning, smart regularisation).
2. **Convert** the improved model to **ONNX** using YOLO26's NMS-free end-to-end export, so the deployed model is production-ready and framework-agnostic.
3. **Containerise** the inference logic in a Docker image with a **fixed, standardised CLI** that the instructor will run on an unseen holdout set during the class activity. Your image will be scored on mAP@0.5 and ranked on a live leaderboard.

The ranking is not the point — the point is that you ship a real, reproducible, frozen artefact that runs the same way on any machine. That is the central skill of model deployment.

The dataset is the same as Week 1:

- [Cat Detection Dataset on Google Drive](https://drive.google.com/drive/folders/1qeGvkaK7UkNMYoESQHxGbV4DRH8EgEb0?usp=drive_link)

## Learning Goals

You will demonstrate that you can:

- Improve a real detector with **CNN-relevant techniques** from Week 2 (transfer learning across model sizes, data augmentation, regularisation, learning-rate schedules).
- **Export** a trained PyTorch model to **ONNX** and **load it in `onnxruntime`** for framework-independent inference.
- Build a small Python inference module around the ONNX model that produces **bounding-box detections** (not just labels).
- **Containerise** the inference with a clean, reproducible **Dockerfile** and ship it to a public registry (Docker Hub or GHCR).
- Conform to a **fixed, standardised CLI** so the instructor can run every student's container the same way.

## Prerequisites

- Completed the Week-1 assessment (`m6-04-assessment`).
- Docker installed and a free [Docker Hub](https://hub.docker.com/) (or GitHub Container Registry) account.
- The lessons and labs from days 6–8 of Unit 6, in particular Day 08 — **Model Conversions & Inferencing**.

```bash
pip install ultralytics onnx onnxruntime numpy pillow opencv-python
```

## Why YOLO26 Makes This Manageable

The Week-1 assessment had you train YOLO26 in PyTorch. YOLO26's **end-to-end NMS-free** design means its ONNX export is unusually clean: the model directly outputs final detections (no separate NMS post-processing required). One-line export, one straightforward inference function, and you're done with the conversion step.

```python
from ultralytics import YOLO
model = YOLO("runs/cats_v2/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=17)
# -> runs/cats_v2/weights/best.onnx
```

You'll then load that `.onnx` file in `onnxruntime` (a tiny CPU-friendly runtime) and run it from your container.

## Part A — Improve the Detector (in a notebook)

Open `m6-09-assessment.ipynb` at the root of your repository.

### 1. Recap

In a markdown cell, briefly recap your Week-1 result:
- Best validation mAP@0.5 and mAP@0.5:0.95 from `m6-04-assessment`.
- Two specific weaknesses you observed in the failure cases.

### 2. Pick **at least three** Week-2 techniques

Apply at least three of the following — the bigger and more diverse, the better:

- **Different YOLO26 variant** — re-fine-tune from a different size (`yolo26n` / `s` / `m` / `l` / `x`) than the one you picked in Week 1. Going *up* a size (e.g. `n → s` or `s → m`) is the obvious move if your Week-1 run looked under-fit; going *down* a size with stronger augmentation can also be a smart trade if you want a smaller image to ship.
- **Stronger augmentation** — turn on / tune Ultralytics' `mosaic`, `mixup`, `copy_paste`, `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`, `flipud`, `fliplr`.
- **Longer training + cosine schedule** — increase epochs (e.g. 60–100) and use the cosine LR schedule (`cos_lr=True`).
- **Two-stage transfer learning** — train the head only for a few epochs, then unfreeze the backbone with a smaller LR.
- **Better regularisation** — add `weight_decay`, tune `dropout` if your variant supports it, use early stopping (`patience=...`).
- **More data discipline** — fix mislabelled / low-quality images you spotted in Week 1; balance class counts if uneven.

For each technique you apply, the notebook must contain:
- A short markdown explanation of *why* you're trying it.
- A code cell with the actual training run.
- The training/validation curves and final test-set metrics.

### 3. Compare against your Week-1 baseline

Produce a comparison table:

| Run | Backbone | Tricks | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|---|
| Week-1 baseline | yolo26&lt;your variant&gt; | none | … | … | … | … |
| v2 — run 1 | … | … | … | … | … | … |
| v2 — run 2 | … | … | … | … | … | … |
| **v2 — best** | … | … | … | … | … | … |

Choose your **best** run as the model you'll ship.

### 4. Export to ONNX

```python
from ultralytics import YOLO
model = YOLO("runs/<your-best-run>/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=17, dynamic=False)
```

Sanity-check the export by:
- Loading `best.onnx` with `onnxruntime` and running inference on a handful of test images.
- Confirming that the boxes from the ONNX model match the boxes from the original PyTorch model (within tiny numerical tolerance).

## Part B — Containerise the Inference

Now you'll build a Docker image that wraps your ONNX model with a fixed CLI that the instructor can run on **unseen images**. Every student's container will follow exactly the same interface so the leaderboard run is fully automated.

### B.1 Repository layout

Inside your repo, create a `container/` directory that looks like:

```
container/
  Dockerfile
  STUDENT.json
  requirements.txt
  app/
    __init__.py
    cli.py
    detector.py
  models/
    best.onnx
```

### B.2 STUDENT.json (required)

This single file is how the instructor identifies you on the leaderboard. Place it at `container/STUDENT.json` and **inside the image at `/app/STUDENT.json`** (the Dockerfile will copy it there). Schema — exact field names, all required:

```json
{
  "first_name": "Alice",
  "last_name": "Garcia",
  "team": "alice-garcia",
  "model": {
    "framework": "yolo26",
    "variant": "yolo26s",
    "imgsz": 640,
    "epochs_total": 80,
    "tricks": ["mosaic", "cos_lr", "two_stage_finetune"]
  },
  "notes": "anything you want — short!"
}
```

`first_name` and `last_name` are mandatory and used to populate the leaderboard. `team` is a single lowercase-with-dashes slug used as the row key (use `firstname-lastname` if you have no team name).

### B.3 The standardised CLI

Your image **must** support exactly two subcommands. The instructor will run them with `docker run`. The container must also have **`python /app/cli.py`** (or the equivalent entrypoint script) defined.

#### `info`

Prints `STUDENT.json` to stdout. Used by the leaderboard runner to register your entry.

```bash
docker run --rm <your-image> info
```

Expected output: the contents of `/app/STUDENT.json`, valid JSON, on stdout, exit code 0.

#### `predict`

Runs your ONNX model on a folder of images and writes a CSV of bounding-box predictions to a fixed path.

```bash
docker run --rm \
  -v /absolute/path/to/holdout:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  <your-image> predict
```

- **Input** — `/data/input/` will contain image files (`.jpg`, `.jpeg`, `.png`). Filenames are arbitrary; treat the path **relative to `/data/input/`** as the image identifier (subdirectories are possible — preserve the relative path).
- **Output** — write **exactly** `/data/output/predictions.csv` with the schema below. Overwrite if it already exists. Exit code 0 on success.

#### Output CSV schema

`/data/output/predictions.csv` — UTF-8, comma-separated, **with header**:

```
image_path,xmin,ymin,xmax,ymax,confidence,class
```

- `image_path` — path **relative to `/data/input/`** (e.g. `img_017.jpg` or `subdir/img_017.jpg`). Use forward slashes.
- `xmin, ymin, xmax, ymax` — **absolute pixel coordinates** of the bounding-box corners in the **original image** (top-left origin = `(0, 0)`). Floats are fine; clip to image bounds.
- `confidence` — float in `[0, 1]`, the detection score.
- `class` — class **name** as a string, matching the names in your `data.yaml` (e.g. `cat`).
- One row per detected box. **Multiple boxes per image** are expected; just write multiple rows with the same `image_path`.
- For images with **no detections**, write a single row with the `image_path` filled in and the other six fields empty:

  ```csv
  empty_img.jpg,,,,,,
  ```

Example:

```csv
image_path,xmin,ymin,xmax,ymax,confidence,class
img_001.jpg,123.4,55.0,478.2,401.7,0.91,cat
img_001.jpg,512.0,200.5,640.0,330.1,0.74,cat
img_002.jpg,,,,,,
subdir/img_003.jpg,40.1,12.0,300.5,250.0,0.88,cat
```

Stick to this format exactly. The leaderboard scoring script depends on it.

### B.4 Dockerfile

Use a small Python base image. Keep the image lean — install `onnxruntime` (CPU build), not the full `ultralytics` package, in the runtime image. Reference (your image is welcome to differ; this is a known-good starting point):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY container/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY container/app /app/app
COPY container/models /app/models
COPY container/STUDENT.json /app/STUDENT.json

ENTRYPOINT ["python", "/app/app/cli.py"]
```

`requirements.txt` minimal:

```
onnxruntime==1.18.0
numpy
pillow
opencv-python-headless
```

(Pin versions you actually tested with.)

### B.5 Inference logic

Inside `app/detector.py`, implement a class that loads `models/best.onnx` once and exposes a `predict(image_path) -> list[dict]` method returning the raw boxes (with original-image-pixel coordinates). Inside `app/cli.py`, parse the subcommand (`info` / `predict`), iterate over `/data/input/`, and write the CSV described above.

Sketch:

```python
# app/detector.py
import numpy as np, onnxruntime as ort
from PIL import Image

class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # letterbox to (imgsz, imgsz), preserve aspect ratio with padding,
        # remember scale + pad so we can map predictions back to original pixels
        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        out = self.session.run(None, {self.input_name: x})[0]  # YOLO26 e2e: (1, 300, 6)
        out = out[0]  # (300, 6) -> [x1, y1, x2, y2, score, class]

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
            # undo letterbox (input-space pixels -> original-image pixels)
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            # clip to image bounds
            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))
            results.append({
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
```

> The exact YOLO26 ONNX output shape is `(N, 300, 6)` for the default end-to-end head — six values per detection: `[x1, y1, x2, y2, score, class]`, max 300 detections per image. Confirm the shape with `session.get_outputs()[0].shape` before assuming it. If you exported with `end2end=False` you'll get the legacy `(N, nc + 4, 8400)` shape and will need to add NMS yourself — strongly recommended to stick with the default end-to-end export.

### B.6 Build, test, and publish

1. Build:

   ```bash
   docker build -t <dockerhub-username>/cat-detector:final -f container/Dockerfile .
   ```

2. Test locally — create a small folder of test images and confirm both subcommands work:

   ```bash
   docker run --rm <dockerhub-username>/cat-detector:final info

   mkdir -p /tmp/inp /tmp/out
   cp some/test/images/*.jpg /tmp/inp/
   docker run --rm \
     -v /tmp/inp:/data/input:ro \
     -v /tmp/out:/data/output \
     <dockerhub-username>/cat-detector:final predict

   cat /tmp/out/predictions.csv | head
   ```

3. Push:

   ```bash
   docker login
   docker push <dockerhub-username>/cat-detector:final
   ```

4. Verify the image works on a clean machine by pulling and running it as if you were the instructor.

## Submission

### What to submit

A pull request on your fork containing:

- `m6-09-assessment.ipynb` — the improvement notebook with comparison table.
- `container/` — the full container source (Dockerfile, app code, STUDENT.json, models folder placeholder; **the actual `best.onnx` does not need to be committed if it is large** — it must be inside the published image).
- `runs/<your-best-run>/weights/best.pt` — the PyTorch checkpoint (or a Drive link if too large).
- `README.md` at the repo root containing **the exact `docker pull` command and image tag** the instructor should use, e.g.:

  ```text
  ## Image for leaderboard
  docker pull alicegarcia/cat-detector:final
  Image: alicegarcia/cat-detector:final
  Student: Alice Garcia
  ```

### Definition of done (checklist)

- [ ] At least 3 Week-2 techniques applied; comparison table vs Week-1 baseline.
- [ ] Best model exported to ONNX; ONNX-vs-PyTorch sanity check shown in the notebook.
- [ ] `STUDENT.json` valid against the schema, with real first/last names.
- [ ] Container builds cleanly from `Dockerfile`.
- [ ] `info` subcommand prints the JSON student record.
- [ ] `predict` subcommand reads `/data/input/`, writes `/data/output/predictions.csv` with the exact schema above.
- [ ] CSV verified on a small local folder of test images.
- [ ] Image pushed to a public registry; pull command included in the repo README.

### How to submit

Paste the link to your Pull Request in the Student Portal.

## Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| Improvement over baseline | 25% | Clear, well-justified Week-2 techniques applied; measurable mAP@0.5 lift on the test split |
| ONNX conversion correctness | 15% | Clean export; ONNX inference matches PyTorch within tolerance |
| Container compliance | 25% | Image follows the standardised CLI and CSV schema **exactly** — this is what makes the leaderboard run possible |
| Reproducibility | 15% | Pinned versions, working pull command, clean repo |
| Code quality & communication | 10% | Readable code, useful comments where intent is non-obvious, sensible repo layout |
| Leaderboard performance | 10% | Bonus weight for relative mAP@0.5 ranking on the unseen instructor holdout |

> If your container does not run, or the CSV does not parse, the leaderboard run is skipped — but the rest of the rubric still applies. Compliance > performance.

## What Happens on Friday's Class Activity

During the class activity period the instructor will:

1. Pull every student's image from the registry.
2. Run `docker run --rm <image> info` to populate the leaderboard with `first_name + last_name`.
3. Run `docker run --rm -v /unseen/holdout:/data/input:ro -v /tmp/<student>/out:/data/output <image> predict` for each image.
4. Score the resulting `predictions.csv` against the held-out ground truth using **mAP@0.5** as the primary metric, with mean inference time per image as the tiebreaker.
5. Publish a live leaderboard table.

The activity is collaborative — bring questions, watch how others approached the problem, and look for design choices that translate to your future production work.

## Image for leaderboard

```text
docker pull seljankhasiyeva/cat-detector:final
Image: seljankhasiyeva/cat-detector:final
Student: Seljan Khasiyeva
```