"""Part 3: Text Recognition (frozen PaddleOCR recognizer).

Pipeline position:  [Detection] -> [Image Cropper] -> [THIS: Recognition] -> [LLM]
INPUT : folder with cropped word images (+ manifest.json from the cropper)
OUTPUT: recognized.json (text + score + coordinates)  and  recognized.txt (plain text)

The recognizer is FROZEN: no training, inference only.
Info for my colab setup:  !pip install paddlepaddle paddleocr   (GPU: paddlepaddle-gpu)
"""
import json
from pathlib import Path

CROPS_DIR = "crops"          # output folder of the cropper (Dennis)
OUT_DIR = "recog_out"
MODEL_NAME = "PP-OCRv5_server_rec"   # check: same model family as the detector


# ---------- Step 1: load the frozen model ----------
def load_recognizer(model_name=MODEL_NAME):
    from paddleocr import TextRecognition
    return TextRecognition(model_name=model_name)


# ---------- Step 2: read the cropper output ----------
def load_items(crops_dir):
    """Returns a list of dicts with at least 'file'. Uses manifest.json if present."""
    crops_dir = Path(crops_dir)
    manifest = crops_dir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    # fallback: no coordinates, just all images
    return [{"file": p.name} for p in sorted(crops_dir.glob("*.png"))]


# ---------- Step 3: recognize ONE image ----------
def recognize_one(model, image_path):
    """Returns (text, score) for a single cropped word image."""
    res = model.predict(input=str(image_path), batch_size=1)[0]
    return res["rec_text"], float(res["rec_score"])


# ---------- Step 4: recognize ALL crops ----------
def recognize_all(model, crops_dir, items):
    for it in items:
        text, score = recognize_one(model, Path(crops_dir) / it["file"])
        it["text"], it["score"] = text, score
    return items


# ---------- Step 5: simple reading order (top->bottom, left->right) ----------
def sort_reading_order(items, line_tol=0.6):
    """Group words into lines by y-center, then sort each line by x-center.
    line_tol: a word joins a line if |cy - line_cy| < line_tol * word height."""
    items = [i for i in items if "cy" in i]
    items.sort(key=lambda i: i["cy"])
    lines = []
    for it in items:
        h = max(it["box"][3] - it["box"][1], 1)
        for line in lines:
            if abs(it["cy"] - line["cy"]) < line_tol * h:
                line["words"].append(it)
                line["cy"] = sum(w["cy"] for w in line["words"]) / len(line["words"])
                break
        else:
            lines.append({"cy": it["cy"], "words": [it]})
    return [sorted(l["words"], key=lambda w: w["cx"]) for l in lines]


# ---------- Step 6: write the outputs ----------
def write_outputs(items, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recognized.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    if all("cy" in i for i in items):
        text = "\n".join(" ".join(w["text"] for w in line) for line in sort_reading_order(items))
    else:  # no coordinates -> keep file order
        text = "\n".join(i["text"] for i in items)
    (out_dir / "recognized.txt").write_text(text + "\n")
    return text


def main():
    model = load_recognizer()                                # Step 1
    items = load_items(CROPS_DIR)                            # Step 2
    items = recognize_all(model, CROPS_DIR, items)           # Steps 3+4
    text = write_outputs(items, OUT_DIR)                     # Steps 5+6
    print(text)


if __name__ == "__main__":
    main()