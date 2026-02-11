# Multimodal Token Cost Calculator

A Python CLI that calculates token counts and estimated costs for text, images, CSV, and PDF files. Tracks stats (resolution, dimensions, character count, file size) and reports token counts and costs across **GPT-4.1**, **Gemini**, and **Claude Haiku 4.5** (Bedrock).

## Summary

- **Stats per file** — resolution, dimensions, character count, file size, format
- **Token estimates** — GPT-4.1 (local), Gemini (API), Haiku 4.5 (Bedrock API)
- **Cost estimates** — USD using provider pricing tables

## Supported Modalities

| Modality | Extensions | Stats |
|----------|------------|-------|
| **Text** | `.txt`, `.md`, `.json`, `.xml`, `.html` | character_count, word_count, line_count, file_size_bytes |
| **CSV** | `.csv`, `.tsv` | Same as text (tabular data) |
| **Image** | `.jpg`, `.png`, `.webp`, `.gif` | width, height, format, file_size_bytes, DPI (if available) |
| **PDF** | `.pdf` | page_count, page_dimensions, extracted_char_count, file_size_bytes |

## Installation

```bash
pip install -r requirements.txt
```

### Optional (for Gemini and Bedrock token counts)

```bash
pip install google-genai   # Gemini API
pip install boto3         # AWS Bedrock
```

## Setup

Create a `.env` file for API keys (loaded automatically):

```
GOOGLE_API_KEY=your_gemini_key
# or
GEMINI_API_KEY=your_gemini_key
```

For Bedrock (Haiku 4.5), configure AWS credentials (`~/.aws/credentials`, `AWS_ACCESS_KEY_ID`, or IAM role).

## Usage

```bash
# Single file
python cost_calculator.py sample/bank-statement.txt

# Directory (recursive by default)
python cost_calculator.py sample/

# Output formats
python cost_calculator.py sample/ --format table   # Human-readable (default)
python cost_calculator.py sample/ --format json    # Machine-readable
python cost_calculator.py sample/ --format both    # Table + JSON

# Options
python cost_calculator.py sample/ --model gpt-4.1  # Cost model (default)
python cost_calculator.py sample/ --detail low      # Low-detail images (85 tokens)
python cost_calculator.py sample/ --no-recursive   # Don't recurse into subdirs
```

## Sample Output

![Token cost calculator output](results.png)

- **GPT-4.1** — Local estimates (always shown)
- **Gemini** — Via `count_tokens` API (requires `GOOGLE_API_KEY`)
- **Haiku-4.5** — Via Bedrock `CountTokens` (requires boto3 + AWS credentials)
- `-` means the API was not available (missing key/creds or error)

## Token Calculation

### GPT-4.1 (OpenAI family)

- **Text/CSV:** `tiktoken` with `o200k_base`
- **Images (high detail):** Scale to fit 2048×2048 → shortest side 768px → 512×512 tiles → `85 + 170 × tiles`
- **Images (low detail):** 85 tokens fixed
- **PDF:** Text tokens + image tokens per page

### Gemini

- Uses Google `count_tokens` API for text, images, and PDFs

### Claude Haiku 4.5 (Bedrock)

- Uses AWS Bedrock `CountTokens` API with Converse format

## ⚠️ Caveats

**These token and cost calculations are estimates and may not match official provider metrics.**

- **GPT-4.1 (vision):** The image token formula (85 base + 170 per 512×512 tile) comes from community sources and pricing examples. Azure/OpenAI do not publish a complete step-by-step algorithm; official docs describe the concept and point to pricing calculators. For authoritative counts, use the [Azure Pricing Calculator](https://azure.microsoft.com/en-in/pricing/details/cognitive-services/openai-service/) or API response metadata.
- **Gemini:** Uses Google’s `count_tokens` API and is generally reliable.
- **Bedrock (Haiku):** Uses AWS `CountTokens` API and reflects actual API usage.

When billing matters, rely on official tools (pricing calculators, token metrics, API response metadata) rather than this script’s estimates.

## Project Structure

```
llm-tokens/
├── cost_calculator.py    # Main CLI
├── llm_tokens/           # Package
│   ├── stats.py          # Stats extraction (PIL, PyMuPDF)
│   ├── tokens.py         # Token logic (tiktoken, OpenAI image formula)
│   ├── cost.py           # Cost from tokens + pricing
│   └── pricing.py       # Provider pricing tables
├── requirements.txt
└── sample/               # Sample files
```

## Dependencies

- `tiktoken` — Text tokenization
- `Pillow` — Image dimensions and format
- `PyMuPDF` — PDF page count, dimensions, text extraction
- `tokencost` — Cost calculation (400+ models)
- `python-dotenv` — Load `.env` for API keys
