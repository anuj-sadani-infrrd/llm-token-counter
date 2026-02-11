#!/usr/bin/env python3
"""
Multimodal Token Cost Calculator

Calculates token counts and estimated costs for text, image, and PDF files.
Tracks stats: resolution, dimensions, character count, etc.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llm_tokens.stats import discover_files, get_file_stats
from llm_tokens.tokens import estimate_tokens
from llm_tokens.cost import estimate_cost


def format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def format_cost(c: float) -> str:
    """Format cost as USD."""
    if c == 0:
        return "$0"
    if c < 0.0001:
        return f"${c:.6f}"
    if c < 0.01:
        return f"${c:.4f}"
    return f"${c:.4f}"


def format_file_size(n: int) -> str:
    """Format file size for display (e.g., 4.9K, 181K, 1.2M)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}K"
    return f"{n / (1024 * 1024):.1f}M"


def _rel_path(path: Path) -> str:
    """Return path relative to cwd if possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _get_google_api_key() -> str | None:
    """Get Google/Gemini API key from env (loaded from .env by dotenv)."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


BEDROCK_HAIKU_45_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"


def _count_tokens_bedrock(path: Path, stats) -> tuple[int | None, str | None]:
    """
    Use AWS Bedrock CountTokens API for Claude Haiku 4.5.
    Returns (token_count, error_message). Needs AWS credentials configured.
    """
    try:
        import boto3
    except ImportError:
        return None, "boto3 required for Bedrock (pip install boto3)"

    try:
        client = boto3.client("bedrock-runtime")
    except Exception as e:
        return None, f"AWS credentials/region: {e}"

    try:
        if stats.modality in ("text", "csv") and stats.text_stats:
            content_blocks = [{"text": path.read_text(encoding=stats.text_stats.encoding or "utf-8", errors="replace")}]
        elif stats.modality == "image":
            ext = path.suffix.lower()
            fmt = "png" if ext == ".png" else "jpeg" if ext in (".jpg", ".jpeg") else "png"
            content_blocks = [
                {"image": {"format": fmt, "source": {"bytes": path.read_bytes()}}}
            ]
        elif stats.modality == "pdf":
            # Bedrock Converse expects images; render first PDF page to PNG
            import fitz
            doc = fitz.open(path)
            if len(doc) == 0:
                doc.close()
                return None, "Empty PDF"
            page = doc[0]
            img_bytes = page.get_pixmap(dpi=150).tobytes()
            doc.close()
            content_blocks = [{"image": {"format": "png", "source": {"bytes": img_bytes}}}]
        else:
            return None, "Unsupported modality"

        input_converse = {
            "messages": [
                {"role": "user", "content": content_blocks}
            ]
        }

        resp = client.count_tokens(
            modelId=BEDROCK_HAIKU_45_MODEL,
            input={"converse": input_converse},
        )
        return resp.get("inputTokens", 0), None
    except Exception as e:
        return None, str(e)


def _count_tokens_google(path: Path, stats) -> tuple[int | None, str | None]:
    """
    Use Google GenAI count_tokens API for text, images, and PDFs.
    Returns (token_count, error_message). error_message is set on failure.
    """
    api_key = _get_google_api_key()
    if not api_key:
        return None, "Missing API key (set GOOGLE_API_KEY or GEMINI_API_KEY in .env)"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        if stats.modality in ("text", "csv") and stats.text_stats:
            content = path.read_text(encoding=stats.text_stats.encoding or "utf-8", errors="replace")
            contents = [content]
        else:
            ext = path.suffix.lower()
            if ext == ".pdf":
                mime = "application/pdf"
            else:
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
                mime = mime_map.get(ext, "image/jpeg")
            part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
            contents = [part]

        resp = client.models.count_tokens(
            model="gemini-2.0-flash",
            contents=contents,
        )
        return resp.total_tokens, None
    except Exception as e:
        return None, str(e)


def run_single(path: Path, detail: str) -> dict | None:
    """Process a single file and return result dict with GPT, Gemini, and Haiku token counts."""
    stats = get_file_stats(path)
    if not stats:
        return None

    tokens_openai = estimate_tokens(stats, detail=detail, model="gpt-4.1")
    tokens_gemini: int | None = None
    tokens_bedrock: int | None = None

    api_tokens, error = _count_tokens_google(path, stats)
    if error:
        print(f"Warning: Gemini API failed for {path.name}: {error}", file=sys.stderr)
    else:
        tokens_gemini = api_tokens

    api_tokens, error = _count_tokens_bedrock(path, stats)
    if error:
        print(f"Warning: Bedrock API failed for {path.name}: {error}", file=sys.stderr)
    else:
        tokens_bedrock = api_tokens

    cost_openai = estimate_cost(tokens_openai, "gpt-4.1")
    cost_gemini = estimate_cost(tokens_gemini, "gemini-2.0-flash") if tokens_gemini is not None else None
    cost_bedrock = estimate_cost(tokens_bedrock, "bedrock/claude-haiku-4-5") if tokens_bedrock is not None else None

    # Chars: for text = count, for pdf = count, for image = "-"
    char_count = None
    if stats.modality in ("text", "csv") and stats.text_stats:
        char_count = stats.text_stats.character_count
        chars_display = format_number(char_count)
    elif stats.modality == "pdf" and stats.pdf_stats:
        char_count = stats.pdf_stats.extracted_char_count
        chars_display = format_number(char_count)
    else:
        chars_display = "-"

    dims = stats.dimensions_str

    return {
        "file": _rel_path(path),
        "modality": stats.modality,
        "chars_display": chars_display,
        "char_count": char_count,
        "dims": dims,
        "file_size_bytes": stats.file_size_bytes,
        "tokens_openai": tokens_openai,
        "tokens_gemini": tokens_gemini,
        "tokens_bedrock": tokens_bedrock,
        "cost_openai": cost_openai,
        "cost_gemini": cost_gemini,
        "cost_bedrock": cost_bedrock,
    }


def output_table(results: list[dict]) -> None:
    """Print results as a table with GPT, Gemini, and Haiku token counts."""
    if not results:
        print("No supported files found.")
        return

    headers = [
        "File", "Modality", "Chars", "Dims", "Size",
        "Tokens (GPT-4.1)", "Tokens (Gemini)", "Tokens (Haiku-4.5)",
        "Cost (GPT-4.1)", "Cost (Gemini)", "Cost (Haiku-4.5)",
    ]

    rows = []
    for r in results:
        rows.append([
            r["file"],
            r["modality"],
            str(r["chars_display"]),
            r["dims"],
            format_file_size(r["file_size_bytes"]),
            format_number(r["tokens_openai"]),
            format_number(r["tokens_gemini"]) if r.get("tokens_gemini") is not None else "-",
            format_number(r["tokens_bedrock"]) if r.get("tokens_bedrock") is not None else "-",
            format_cost(r["cost_openai"]),
            format_cost(r["cost_gemini"]) if r.get("cost_gemini") is not None else "-",
            format_cost(r["cost_bedrock"]) if r.get("cost_bedrock") is not None else "-",
        ])

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    col_widths[0] = min(col_widths[0], 42)

    def fmt_row(cells: list, widths: list) -> str:
        return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, widths))

    print(fmt_row(headers, col_widths))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for row in rows:
        print(fmt_row(row, col_widths))

    total_openai = sum(r["tokens_openai"] for r in results)
    total_gemini = sum(r["tokens_gemini"] or 0 for r in results)
    total_bedrock = sum(r["tokens_bedrock"] or 0 for r in results)
    total_cost_openai = sum(r["cost_openai"] for r in results)
    total_cost_gemini = sum(r["cost_gemini"] or 0 for r in results)
    total_cost_bedrock = sum(r["cost_bedrock"] or 0 for r in results)
    total_size = sum(r["file_size_bytes"] for r in results)
    total_row = [
        "(total)", "", "", "", format_file_size(total_size),
        format_number(total_openai),
        format_number(total_gemini) if total_gemini else "-",
        format_number(total_bedrock) if total_bedrock else "-",
        format_cost(total_cost_openai),
        format_cost(total_cost_gemini) if total_cost_gemini else "-",
        format_cost(total_cost_bedrock) if total_cost_bedrock else "-",
    ]
    print(fmt_row(total_row, col_widths))


def output_json(results: list[dict]) -> None:
    """Print results as JSON."""
    out = []
    for r in results:
        entry = {
            "file": r["file"],
            "modality": r["modality"],
            "dimensions": r["dims"],
            "file_size_bytes": r["file_size_bytes"],
            "tokens_openai": r["tokens_openai"],
            "tokens_gemini": r.get("tokens_gemini"),
            "tokens_bedrock": r.get("tokens_bedrock"),
            "cost_openai_usd": round(r["cost_openai"], 6),
            "cost_gemini_usd": round(r["cost_gemini"], 6) if r.get("cost_gemini") is not None else None,
            "cost_bedrock_usd": round(r["cost_bedrock"], 6) if r.get("cost_bedrock") is not None else None,
        }
        if r.get("char_count") is not None:
            entry["character_count"] = r["char_count"]
        out.append(entry)
    print(json.dumps(out, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate token counts and costs for text, image, and PDF files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="File(s) or directory(ies) to process",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Model for cost estimation (default: gpt-4.1)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai"],
        help="Provider (default: openai)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "both"],
        default="table",
        help="Output format: table, json, or both (default: table)",
    )
    parser.add_argument(
        "--detail",
        choices=["low", "high"],
        default="high",
        help="Image detail level for token calculation (default: high)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )

    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.paths]
    for p in paths:
        if not p.exists():
            print(f"Error: path does not exist: {p}", file=sys.stderr)
            return 1

    files = discover_files(paths, recursive=not args.no_recursive)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    results = []
    for f in files:
        r = run_single(f, args.detail)
        if r:
            results.append(r)

    if args.format == "json":
        output_json(results)
    elif args.format == "both":
        output_table(results)
        print()
        output_json(results)
    else:
        output_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
