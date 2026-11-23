import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ResumeError(ValueError):
    pass


def extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError) as exc:
        raise ResumeError(f"could not read PDF: {exc}") from exc

    text = clean(text)
    if len(text) < 50:
        # usually a scanned PDF (just images, no text layer)
        raise ResumeError("no text found in PDF - is it a scanned image?")
    return text


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
