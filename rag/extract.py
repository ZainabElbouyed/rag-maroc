"""Étape 3a : extraction du texte des PDF (PyMuPDF, repli pdfplumber, OCR optionnel)."""
import io
import warnings
from pathlib import Path


def _open_pymupdf():
    try:
        import pymupdf as fitz  # type: ignore
        return fitz
    except ImportError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import fitz  # type: ignore
        return fitz
    except ImportError:
        return None


def _ocr_page(page) -> str:
    """OCR d'une page scannée (nécessite Tesseract + pytesseract + Pillow)."""
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            return pytesseract.image_to_string(img, lang="fra")
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(img)
    except Exception as exc:  # noqa: BLE001
        print(f"  [OCR indisponible] {exc.__class__.__name__}: {exc}")
        return ""


def extract_pdf(path: Path, ocr: bool = True) -> tuple[list[str], int]:
    """Retourne (texte de chaque page, nombre de pages passées par l'OCR)."""
    path = Path(path)
    fitz = _open_pymupdf()
    if fitz is not None:
        pages, n_ocr = [], 0
        with fitz.open(str(path)) as doc:
            for page in doc:
                txt = page.get_text("text") or ""
                if ocr and len(txt.strip()) < 30:
                    ocr_txt = _ocr_page(page)
                    if ocr_txt.strip():
                        txt, n_ocr = ocr_txt, n_ocr + 1
                pages.append(txt)
        return pages, n_ocr
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("Installez pymupdf ou pdfplumber : pip install pymupdf") from exc
    with pdfplumber.open(str(path)) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages], 0
