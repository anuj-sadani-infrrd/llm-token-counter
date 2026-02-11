"""Token estimation for text, images, and PDFs using OpenAI/GPT-4o logic."""

import math
from pathlib import Path

from llm_tokens.stats import FileStats

# OpenAI GPT-4o/4.1/4.5 image token constants
OPENAI_IMAGE_BASE_TOKENS = 85
OPENAI_IMAGE_TILE_TOKENS = 170
OPENAI_IMAGE_MAX_SIDE = 2048
OPENAI_IMAGE_SHORT_SIDE = 768
OPENAI_IMAGE_TILE_SIZE = 512

# PDF points to pixels at 96 DPI (standard screen)
PDF_POINTS_TO_PX = 96 / 72


def _get_tiktoken_encoder(model: str = "gpt-4.1"):
    """Lazy load tiktoken encoder. GPT-4.1/4o use o200k_base; gpt-4.1 not in tiktoken yet, fallback to o200k_base."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # gpt-4.1 not in tiktoken registry; same family as gpt-4o (4o/4.1/4.5)
        return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in text using tiktoken. GPT-4.1/4o use o200k_base."""
    enc = _get_tiktoken_encoder(model)
    return len(enc.encode(text))


def _openai_image_tokens_high_detail(width: int, height: int) -> int:
    """
    OpenAI high-detail image token calculation (GPT-4o, 4.1, 4.5).
    Per official docs: https://platform.openai.com/docs/guides/vision#calculating-costs
    - Scale to fit within 2048×2048 (scale down only; no resize if already fits)
    - Scale so shortest side = 768px
    - Tiles = ceil(w/512) × ceil(h/512)
    - Tokens = 85 + 170 × tiles
    """
    if width <= 0 or height <= 0:
        return OPENAI_IMAGE_BASE_TOKENS

    # Scale to fit within 2048×2048 (only scale down; don't scale up small images)
    scale1 = min(1.0, OPENAI_IMAGE_MAX_SIDE / width, OPENAI_IMAGE_MAX_SIDE / height)
    w1 = width * scale1
    h1 = height * scale1

    # Scale so shortest side = 768
    short = min(w1, h1)
    if short <= 0:
        return OPENAI_IMAGE_BASE_TOKENS
    scale2 = OPENAI_IMAGE_SHORT_SIDE / short
    w2 = w1 * scale2
    h2 = h1 * scale2

    tiles_w = math.ceil(w2 / OPENAI_IMAGE_TILE_SIZE)
    tiles_h = math.ceil(h2 / OPENAI_IMAGE_TILE_SIZE)
    tiles = tiles_w * tiles_h

    return OPENAI_IMAGE_BASE_TOKENS + (OPENAI_IMAGE_TILE_TOKENS * tiles)


def _openai_image_tokens_low_detail() -> int:
    """OpenAI low-detail: fixed 85 tokens."""
    return OPENAI_IMAGE_BASE_TOKENS


def count_image_tokens_openai(
    width: int,
    height: int,
    detail: str = "high",
) -> int:
    """
    Count image tokens for OpenAI vision models.
    detail: "low" (85 tokens) or "high" (tile-based).
    """
    if detail == "low":
        return _openai_image_tokens_low_detail()
    return _openai_image_tokens_high_detail(width, height)


def _pdf_page_to_pixels(width_pt: float, height_pt: float) -> tuple[int, int]:
    """Convert PDF page dimensions (points) to pixel equivalent at 96 DPI."""
    w_px = int(width_pt * PDF_POINTS_TO_PX)
    h_px = int(height_pt * PDF_POINTS_TO_PX)
    return (max(1, w_px), max(1, h_px))


def count_pdf_tokens_openai(
    page_count: int,
    page_dimensions: list[tuple[float, float]],
    extracted_text: str,
    detail: str = "high",
    model: str = "gpt-4.1",
) -> int:
    """
    Estimate PDF tokens for OpenAI: text tokens + image tokens per page.
    """
    text_tokens = count_text_tokens(extracted_text, model=model)

    image_tokens = 0
    for w_pt, h_pt in page_dimensions:
        w_px, h_px = _pdf_page_to_pixels(w_pt, h_pt)
        image_tokens += count_image_tokens_openai(w_px, h_px, detail)

    return text_tokens + image_tokens


def estimate_tokens(
    stats: FileStats,
    detail: str = "high",
    model: str = "gpt-4.1",
) -> int:
    """
    Estimate total tokens for a file based on its stats. Uses GPT-4.1 logic (same as 4o/4.5).
    """
    if stats.modality in ("text", "csv"):
        if stats.text_stats:
            content = Path(stats.path).read_text(
                encoding=stats.text_stats.encoding or "utf-8",
                errors="replace",
            )
            return count_text_tokens(content, model=model)
        return 0

    if stats.modality == "image" and stats.image_stats:
        return count_image_tokens_openai(
            stats.image_stats.width,
            stats.image_stats.height,
            detail=detail,
        )

    if stats.modality == "pdf" and stats.pdf_stats:
        doc = stats.pdf_stats
        # Extract text from PDF
        import fitz

        text_parts = []
        with fitz.open(stats.path) as doc_fitz:
            for i in range(len(doc_fitz)):
                text_parts.append(doc_fitz[i].get_text())
        extracted_text = "".join(text_parts)
        return count_pdf_tokens_openai(
            page_count=doc.page_count,
            page_dimensions=doc.page_dimensions,
            extracted_text=extracted_text,
            detail=detail,
            model=model,
        )

    return 0
