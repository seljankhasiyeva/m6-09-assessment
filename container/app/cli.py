import argparse
import csv
import os
import sys
from pathlib import Path

APP_DIR      = Path(__file__).resolve().parent         
REPO_ROOT    = APP_DIR.parent                          

STUDENT_JSON = REPO_ROOT / "STUDENT.json"
MODEL_PATH   = REPO_ROOT / "models" / "best.onnx"

INPUT_DIR  = Path(os.environ.get("INPUT_DIR",  "/data/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info() -> None:
    """Print STUDENT.json to stdout and exit 0."""
    if not STUDENT_JSON.exists():
        print(f"ERROR: {STUDENT_JSON} not found inside the image.", file=sys.stderr)
        sys.exit(1)
    print(STUDENT_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def cmd_predict(conf: float = 0.25) -> None:
    """
    Iterate over all images in /data/input/ (recursively),
    run the ONNX detector on each, and write predictions.csv.
    """
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from detector import CatDetector 

    if not INPUT_DIR.exists():
        print(f"ERROR: input directory {INPUT_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                abs_path = Path(root) / fname
                rel_path = abs_path.relative_to(INPUT_DIR).as_posix()
                image_paths.append((rel_path, abs_path))

    if not image_paths:
        print("WARNING: no images found in /data/input/", file=sys.stderr)

    print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr)
    detector = CatDetector(str(MODEL_PATH), conf=conf)

    CSV_FIELDS = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for rel_path, abs_path in image_paths:
            try:
                boxes = detector.predict(str(abs_path))
            except Exception as exc:  
                print(f"WARNING: inference failed for {rel_path}: {exc}", file=sys.stderr)
                boxes = []

            if not boxes:
                writer.writerow({
                    "image_path": rel_path,
                    "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                    "confidence": "", "class": "",
                })
            else:
                for box in boxes:
                    writer.writerow({
                        "image_path": rel_path,
                        "xmin":       box["xmin"],
                        "ymin":       box["ymin"],
                        "xmax":       box["xmax"],
                        "ymax":       box["ymax"],
                        "confidence": box["confidence"],
                        "class":      box["class"],
                    })

    print(f"Done. Predictions written to {OUTPUT_CSV}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing & dispatch
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cat Detector CLI — info | predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print STUDENT.json to stdout.")

    predict_parser = sub.add_parser(
        "predict",
        help="Run model on /data/input/, write /data/output/predictions.csv.",
    )
    predict_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "info":
        cmd_info()
    elif args.command == "predict":
        cmd_predict(conf=args.conf)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()