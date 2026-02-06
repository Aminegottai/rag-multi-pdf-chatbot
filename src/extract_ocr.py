import os, json, re
import fitz
import pytesseract
from PIL import Image
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_DIR = "data/pdfs"
OUT_PATH = "data/processed/pages.jsonl"
os.makedirs("data/processed", exist_ok=True)

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th)

def ocr_page(pil_img: Image.Image) -> str:
    pre = preprocess_for_ocr(pil_img)
    config = r"--oem 3 --psm 6"
    return pytesseract.image_to_string(pre, lang="eng", config=config)

def should_do_ocr(extracted_text: str) -> bool:
    return len(extracted_text.strip()) < 200

def source_name(pdf_file: str) -> str:
    return os.path.splitext(os.path.basename(pdf_file))[0].lower()

def main():
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            src = source_name(pdf_file)

            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page = doc[i]
                text = clean_text(page.get_text("text") or "")

                ocr_text = ""
                if should_do_ocr(text):
                    pix = page.get_pixmap(dpi=200)
                    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = clean_text(ocr_page(pil_img))

                merged = text
                if ocr_text:
                    merged = (merged + "\n\n[OCR]\n" + ocr_text).strip()

                rec = {"source": src, "pdf_file": pdf_file, "page": i + 1, "text": merged}
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()