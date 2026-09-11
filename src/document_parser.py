"""
Document parser module for ResearchLens AI.
Extracts and cleans text from PDF, DOCX, and TXT files.
Detects scanned/image-only PDFs and calculates document metrics.
"""

import io
import re
from typing import Dict, Any, Union, BinaryIO
from pypdf import PdfReader
from docx import Document

from src.utils import get_file_extension, estimate_token_count


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted document text:
    - Remove null bytes and non-printable control characters (except tabs and newlines)
    - Normalize Unicode dashes, hyphens, and quotation marks
    - Condense multiple horizontal spaces into a single space
    - Condense excessive blank lines (more than two consecutive newlines)
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Remove null bytes and non-printable chars
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize unicode quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")

    # Normalize horizontal spaces
    text = re.sub(r"[ \t\f\v]+", " ", text)

    # Normalize line breaks: replace CRLF with LF, strip trailing spaces per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Condense excessive newlines (max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_text(file_obj: Union[BinaryIO, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Extract text from a PDF file using pypdf.
    Detects scanned or image-only PDFs where machine-readable text is missing.
    """
    try:
        if isinstance(file_obj, bytes):
            stream = io.BytesIO(file_obj)
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            stream = io.BytesIO(content)
        else:
            stream = file_obj

        reader = PdfReader(stream)
        num_pages = len(reader.pages)

        if num_pages == 0:
            return {
                "text": "",
                "raw_text": "",
                "page_count": 0,
                "is_scanned": False,
                "error": "The PDF document contains 0 pages."
            }

        page_texts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
            except Exception as pe:
                page_texts.append(f"[Error reading page {i+1}: {pe}]")

        full_raw = "\n\n".join(page_texts).strip()
        cleaned = clean_text(full_raw)

        # Scanned PDF detection heuristic:
        # If total characters across pages is very low (e.g. < 60 chars or average < 25 chars/page)
        char_len = len(cleaned)
        avg_chars_per_page = char_len / num_pages if num_pages > 0 else 0

        is_scanned = False
        scanned_message = ""
        if char_len < 60 or (num_pages > 1 and avg_chars_per_page < 30):
            is_scanned = True
            scanned_message = (
                "This PDF appears to contain little or no machine-readable text. "
                "OCR is not included in MVP."
            )

        return {
            "text": cleaned if not is_scanned else "",
            "raw_text": full_raw,
            "page_count": num_pages,
            "is_scanned": is_scanned,
            "error": scanned_message
        }

    except Exception as e:
        return {
            "text": "",
            "raw_text": "",
            "page_count": 0,
            "is_scanned": False,
            "error": f"Failed to parse PDF document: {str(e)}"
        }


def extract_docx_text(file_obj: Union[BinaryIO, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Extract text from a DOCX document using python-docx.
    Extracts body paragraphs as well as table cells.
    """
    try:
        if isinstance(file_obj, bytes):
            stream = io.BytesIO(file_obj)
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            stream = io.BytesIO(content)
        else:
            stream = file_obj

        doc = Document(stream)

        elements = []
        for paragraph in doc.paragraphs:
            p_text = paragraph.text.strip()
            if p_text:
                elements.append(p_text)

        # Also extract table text to not miss tabular results and baselines
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # Deduplicate adjacent cells in merged rows
                    unique_cells = []
                    for c in row_cells:
                        if not unique_cells or c != unique_cells[-1]:
                            unique_cells.append(c)
                    elements.append(" | ".join(unique_cells))

        full_raw = "\n\n".join(elements).strip()
        cleaned = clean_text(full_raw)

        return {
            "text": cleaned,
            "raw_text": full_raw,
            "page_count": 1,
            "is_scanned": False,
            "error": "" if cleaned else "The DOCX document is empty."
        }

    except Exception as e:
        return {
            "text": "",
            "raw_text": "",
            "page_count": 0,
            "is_scanned": False,
            "error": f"Failed to parse DOCX document: {str(e)}"
        }


def extract_txt_text(file_obj: Union[BinaryIO, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Extract text from a plain text file.
    Attempts UTF-8 first, with fallback to latin-1.
    """
    try:
        if isinstance(file_obj, bytes):
            raw_bytes = file_obj
        elif hasattr(file_obj, "read"):
            raw_bytes = file_obj.read()
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
        else:
            raw_bytes = b""

        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("latin-1", errors="replace")

        cleaned = clean_text(raw_text)

        return {
            "text": cleaned,
            "raw_text": raw_text,
            "page_count": 1,
            "is_scanned": False,
            "error": "" if cleaned else "The text document is empty."
        }

    except Exception as e:
        return {
            "text": "",
            "raw_text": "",
            "page_count": 0,
            "is_scanned": False,
            "error": f"Failed to parse text document: {str(e)}"
        }


def extract_text(file_obj: Any, filename: str) -> Dict[str, Any]:
    """
    Main extraction dispatcher.
    Inspects filename extension and calls the corresponding parser.
    Calculates word count, char count, and estimated tokens.
    """
    ext = get_file_extension(filename)

    if ext == ".pdf":
        result = extract_pdf_text(file_obj)
    elif ext == ".docx":
        result = extract_docx_text(file_obj)
    elif ext == ".txt":
        result = extract_txt_text(file_obj)
    else:
        return {
            "text": "",
            "raw_text": "",
            "filename": filename,
            "file_type": ext,
            "word_count": 0,
            "char_count": 0,
            "token_count": 0,
            "page_count": 0,
            "is_scanned": False,
            "error": f"Unsupported file format '{ext}'. Allowed formats: .pdf, .docx, .txt"
        }

    extracted_text = result.get("text", "")
    words = len(extracted_text.split()) if extracted_text else 0
    chars = len(extracted_text) if extracted_text else 0
    tokens = estimate_token_count(extracted_text)

    return {
        "text": extracted_text,
        "raw_text": result.get("raw_text", ""),
        "filename": filename,
        "file_type": ext.upper().lstrip("."),
        "word_count": words,
        "char_count": chars,
        "token_count": tokens,
        "page_count": result.get("page_count", 0),
        "is_scanned": result.get("is_scanned", False),
        "error": result.get("error", "")
    }
