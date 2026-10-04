"""OpenCV preprocessing + OCR + value extraction for medical-report images.
Requires the Tesseract binary (https://github.com/tesseract-ocr/tesseract)."""
import re, cv2, numpy as np, pytesseract

def preprocess(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if w < 1400:                                            # resize
        img = cv2.resize(img, None, fx=1400 / w, fy=1400 / w, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # grayscale
    gray = cv2.fastNlMeansDenoising(gray, None, 10)         # noise removal
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)   # thresholding
    # region extraction: crop to the block of text
    inv = cv2.bitwise_not(th)
    blob = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=2)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, bw, bh = cv2.boundingRect(np.vstack(cnts))
        th = th[y:y + bh, x:x + bw]
    return th

def run_ocr(img_bytes):
    th = preprocess(img_bytes)
    return pytesseract.image_to_string(th, config="--psm 6"), th

PATTERNS = {
    "age":      r"age\D{0,10}(\d{2,3})",
    "trestbps": r"(?:blood\s*pressure|\bbp\b|trestbps)\D{0,15}(\d{2,3})",
    "chol":     r"(?:cholesterol|\bchol\b)\D{0,15}(\d{2,3})",
    "thalach":  r"(?:max(?:imum)?\.?\s*heart\s*rate|thalach)\D{0,15}(\d{2,3})",
    "oldpeak":  r"(?:st\s*depression|oldpeak)\D{0,15}(\d+(?:\.\d+)?)",
    "fbs_raw":  r"fasting\s*(?:blood\s*)?(?:sugar|glucose)\D{0,15}(\d{2,3})",
}

def extract_values(text):
    t, out = text.lower(), {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, t)
        if m:
            out[key] = float(m.group(1))
    if "fbs_raw" in out:
        out["fbs"] = 1 if out.pop("fbs_raw") > 120 else 0
    if re.search(r"\bfemale\b", t): out["sex"] = 0
    elif re.search(r"\bmale\b", t): out["sex"] = 1
    return out
