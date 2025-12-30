# Image Translation Research Prototype

A research-oriented prototype that translates readable text in images while preserving the original visual appearance. The system uses OCR for text detection, OpenAI API for translation, Stable Diffusion for background inpainting, and deterministic text rendering.

## Architecture

This pipeline follows a strict separation of concerns:

1. **OCR Stage**: Detects and extracts text with bounding boxes, orientation, and grouping
2. **Translation Stage**: Translates text using OpenAI API (literal translation, preserves formatting)
3. **Mask Generation**: Creates pixel-accurate masks from OCR bounding boxes
4. **Inpainting Stage**: Uses Stable Diffusion ONLY to reconstruct background pixels where text existed
5. **Text Rendering**: Deterministically renders translated text (NOT using diffusion)
6. **Composition**: Merges rendered text with cleaned background

## Key Principles

- **Stable Diffusion is NEVER used to generate text** - only for background reconstruction
- **Text rendering is deterministic** - uses graphics libraries, not generative models
- **Modular design** - each stage can be swapped independently
- **Open source priority** - uses free, locally runnable tools where possible

## Inpainting Model Constraints

### Default: Stable Diffusion 1.5 Inpainting
- **Model**: `runwayml/stable-diffusion-inpainting`
- **Why**: More conservative, less prone to hallucination, better for small localized background reconstruction
- **Behavior**: Strict mask-bound inpainting with prompts that explicitly forbid text generation

### Experimental: SDXL Inpainting (Optional)
- **Model**: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- **Usage**: Enable with `--use-sdxl` flag for comparison
- **Purpose**: Research evaluation to compare SD 1.5 vs SDXL results
- **Note**: SDXL is NOT used by default - SD 1.5 is the recommended conservative choice

### Inpainting Behavior Rules
- **Strict mask-bound**: Only masked regions are modified
- **Prompts forbid**: text, letters, numbers, symbols, writing, words, characters, typography, font
- **Purpose**: Background reconstruction ONLY, never text generation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# .env file should contain:
OPENAI_API_KEY=your_openai_api_key_here
```

3. Download Stable Diffusion inpainting model (will be downloaded automatically on first run)

## Usage

```bash
python main.py --input path/to/image.jpg --target_lang Spanish --output output.jpg
```

### Arguments

- `--input`: Path to input image file
- `--target_lang`: Target language for translation (e.g., "Spanish", "French", "German")
- `--output`: Path for output image (default: `output_translated.jpg`)
- `--save_intermediates`: Save intermediate artifacts for debugging (default: True)
- `--inpainting_model`: Stable Diffusion model name (default: `runwayml/stable-diffusion-inpainting`, SD 1.5)
- `--use-sdxl`: Use SDXL inpainting (EXPERIMENTAL, for comparison). Default: False (uses SD 1.5)
- `--log-level`: Set logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--no-log-file`: Disable logging to file (logs only to console)

## Pipeline Stages

### 1. Input Handling
- Loads image and normalizes resolution/color space
- Ensures consistent format for processing

### 2. OCR Stage
- Detects all text regions using PaddleOCR
- Extracts text content, bounding boxes, orientation, and grouping
- Outputs structured JSON metadata

### 3. Translation Stage
- Translates each text segment via OpenAI API
- Preserves punctuation, casing, and line breaks
- Calculates text expansion ratio per segment

### 4. Mask Creation
- Converts OCR bounding boxes into pixel-accurate binary masks
- Expands masks slightly to cover anti-aliased edges
- Supports rotated bounding boxes

### 5. Inpainting Stage
- Runs Stable Diffusion inpainting on masked regions only
- **Default**: SD 1.5 Inpainting (conservative, reliable, less prone to hallucination)
- **Experimental**: SDXL Inpainting (optional, enable with `--use-sdxl` for comparison)
- Uses conservative prompts describing clean background
- **Explicitly forbids**: text, letters, numbers, symbols, writing, words, characters, typography, font
- **Strict mask-bound**: Only masked regions are modified, unmasked pixels remain unchanged

### 6. Text Rendering Stage
- Renders translated text deterministically using Pillow
- Matches original placement, rotation, and approximate style
- Dynamically resizes text to fit bounding boxes

### 7. Final Composition
- Merges rendered text with cleaned image
- Outputs final translated image

## Output Artifacts

When `--save_intermediates` is enabled, the following files are saved:

- `ocr_visualization.jpg`: Visual representation of detected text regions
- `text_mask.png`: Binary mask showing regions to be inpainted
- `inpainted_background.jpg`: Image after text removal (before text reinsertion)
- `ocr_metadata.json`: Structured OCR results with bounding boxes and text

## Limitations

- **Font matching**: Approximate font estimation, not pixel-perfect reconstruction
- **Text expansion**: Long translations may overflow bounding boxes
- **Complex backgrounds**: Inpainting may produce artifacts on textured backgrounds
- **Orientation**: Highly rotated text may have alignment issues
- **Logo translation**: Not supported (logos are not translated)
- **Batch processing**: Single image processing only

## Research Focus

This prototype is designed for experimentation with:

- OCR accuracy vs visual fidelity trade-offs
- Stable Diffusion inpainting quality on various backgrounds
- Text expansion handling strategies
- Human-perceived realism of translated text placement

## Where Stable Diffusion Helps and Fails

### Helps:
- Reconstructing clean backgrounds where text was removed
- Handling complex textures and gradients
- Maintaining visual consistency with surrounding areas

### Fails:
- Cannot be trusted to generate readable text (hence deterministic rendering)
- May introduce artifacts on highly detailed backgrounds
- May struggle with very large masked regions
- Cannot preserve exact color gradients in some cases

## Module Structure

```
image-translation/
├── input_handler.py      # Image loading and normalization
├── ocr_module.py         # Text detection and extraction
├── translation_module.py # OpenAI API translation
├── mask_generator.py     # Mask creation from bounding boxes
├── inpainting_module.py  # Stable Diffusion inpainting
├── text_renderer.py      # Deterministic text rendering
├── pipeline.py           # Main pipeline orchestrator
├── utils.py              # Helper functions
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Future Enhancements

- Evaluation metrics (visual similarity, error categories)
- Failure taxonomy documentation
- Portfolio-ready research writeup structure
- Support for additional OCR engines
- Improved font matching algorithms

