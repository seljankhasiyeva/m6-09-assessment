# app/detector.py
import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

        out_shape = self.session.get_outputs()[0].shape
        print(f"[CatDetector] ONNX output shape: {out_shape}")

    def _letterbox(self, img: Image.Image, imgsz: int):
        orig_w, orig_h = img.size
        scale = min(imgsz / orig_w, imgsz / orig_h)

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        pad_x = (imgsz - new_w) // 2
        pad_y = (imgsz - new_h) // 2

        canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        canvas.paste(img_resized, (pad_x, pad_y))

        return canvas, scale, (pad_x, pad_y)

    def predict(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)

        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        raw = self.session.run(None, {self.input_name: x})[0] 
        out = raw[0]  

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0.0, min(float(orig_w), float(x1)))
            y1 = max(0.0, min(float(orig_h), float(y1)))
            x2 = max(0.0, min(float(orig_w), float(x2)))
            y2 = max(0.0, min(float(orig_h), float(y2)))

            cls_idx = int(cls)
            class_name = (
                self.class_names[cls_idx]
                if cls_idx < len(self.class_names)
                else str(cls_idx)
            )

            results.append({
                "xmin":       x1,
                "ymin":       y1,
                "xmax":       x2,
                "ymax":       y2,
                "confidence": float(score),
                "class":      class_name,
            })

        return results