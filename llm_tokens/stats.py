"""Extract stats (dimensions, character count, etc.) from text, image, and PDF files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Modality type constants
# CSV is a distinct modality - Gemini, Claude support it natively for tabular analysis
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml", ".html", ".htm"}
CSV_EXTENSIONS = {".csv", ".tsv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
PDF_EXTENSION = ".pdf"


@dataclass
class TextStats:
    """Stats for text-based files."""

    character_count: int
    word_count: int
    line_count: int
    file_size_bytes: int
    encoding: str = "utf-8"


@dataclass
class ImageStats:
    """Stats for image files."""

    width: int
    height: int
    format: str
    file_size_bytes: int
    dpi: Optional[tuple[float, float]] = None


@dataclass
class PDFStats:
    """Stats for PDF files."""

    page_count: int
    page_dimensions: list[tuple[float, float]]  # (width, height) in points per page
    extracted_char_count: int
    file_size_bytes: int


@dataclass
class FileStats:
    """Unified stats for any file type."""

    path: Path
    modality: str  # "text", "csv", "image", "pdf"
    file_size_bytes: int
    text_stats: Optional[TextStats] = None
    image_stats: Optional[ImageStats] = None
    pdf_stats: Optional[PDFStats] = None

    @property
    def character_count(self) -> Optional[int]:
        if self.text_stats:
            return self.text_stats.character_count
        if self.pdf_stats:
            return self.pdf_stats.extracted_char_count
        return None

    @property
    def dimensions_str(self) -> str:
        if self.image_stats:
            return f"{self.image_stats.width}×{self.image_stats.height}"
        if self.pdf_stats:
            if self.pdf_stats.page_count == 1 and self.pdf_stats.page_dimensions:
                w, h = self.pdf_stats.page_dimensions[0]
                return f"{int(w)}×{int(h)} pts"
            return f"{self.pdf_stats.page_count} pages"
        return "-"


def _get_modality(path: Path) -> Optional[str]:
    """Determine file modality from extension."""
    ext = path.suffix.lower()
    if ext in CSV_EXTENSIONS:
        return "csv"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == PDF_EXTENSION:
        return "pdf"
    return None


def _get_text_stats(path: Path) -> TextStats:
    """Extract stats from a text file."""
    file_size = path.stat().st_size
    encoding = "utf-8"
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
            encoding = "latin-1"
        except Exception:
            content = path.read_bytes().decode("utf-8", errors="replace")
            encoding = "utf-8 (replace)"

    char_count = len(content)
    word_count = len(content.split()) if content.strip() else 0
    line_count = len(content.splitlines()) if content else 0

    return TextStats(
        character_count=char_count,
        word_count=word_count,
        line_count=line_count,
        file_size_bytes=file_size,
        encoding=encoding,
    )


def _get_image_stats(path: Path) -> ImageStats:
    """Extract stats from an image file using Pillow."""
    from PIL import Image

    file_size = path.stat().st_size
    with Image.open(path) as img:
        width, height = img.size
        fmt = img.format or path.suffix.lstrip(".").upper()
        dpi = None
        if "dpi" in img.info:
            dpi = img.info["dpi"]
            if isinstance(dpi, tuple) and len(dpi) >= 2:
                dpi = (float(dpi[0]), float(dpi[1]))

    return ImageStats(
        width=width,
        height=height,
        format=fmt,
        file_size_bytes=file_size,
        dpi=dpi,
    )


def _get_pdf_stats(path: Path) -> PDFStats:
    """Extract stats from a PDF file using PyMuPDF."""
    import fitz

    file_size = path.stat().st_size
    doc = fitz.open(path)
    try:
        page_count = len(doc)
        page_dims = []
        total_chars = 0

        for i in range(page_count):
            page = doc[i]
            rect = page.rect
            page_dims.append((rect.width, rect.height))
            total_chars += len(page.get_text())

        return PDFStats(
            page_count=page_count,
            page_dimensions=page_dims,
            extracted_char_count=total_chars,
            file_size_bytes=file_size,
        )
    finally:
        doc.close()


def get_file_stats(path: Path) -> Optional[FileStats]:
    """
    Extract stats from a file. Returns None if the file type is unsupported.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None

    modality = _get_modality(path)
    if not modality:
        return None

    file_size = path.stat().st_size

    # CSV uses same stats as text (tabular data treated as text for tokenization)
    if modality in ("text", "csv"):
        text_stats = _get_text_stats(path)
        return FileStats(
            path=path,
            modality=modality,
            file_size_bytes=file_size,
            text_stats=text_stats,
        )

    if modality == "image":
        image_stats = _get_image_stats(path)
        return FileStats(
            path=path,
            modality=modality,
            file_size_bytes=file_size,
            image_stats=image_stats,
        )

    if modality == "pdf":
        pdf_stats = _get_pdf_stats(path)
        return FileStats(
            path=path,
            modality=modality,
            file_size_bytes=file_size,
            pdf_stats=pdf_stats,
        )

    return None


def discover_files(paths: list[Path], recursive: bool = True) -> list[Path]:
    """Discover supported files from given paths (files or directories)."""
    supported_ext = TEXT_EXTENSIONS | CSV_EXTENSIONS | IMAGE_EXTENSIONS | {PDF_EXTENSION}
    result = []

    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in supported_ext:
                result.append(path.resolve())
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in path.glob(pattern):
                if f.is_file() and f.suffix.lower() in supported_ext:
                    result.append(f.resolve())

    return sorted(result)
