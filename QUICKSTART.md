# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inpainting, but CPU works too)
- OpenAI API key (already configured in `.env`)

## Step 1: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Note**: This will download:
- PaddleOCR models (for OCR)
- Stable Diffusion 1.5 Inpainting model (~4GB, downloaded on first run)
- Other dependencies

## Step 2: Verify Environment

Your `.env` file already contains the OpenAI API key. Make sure it's valid.

## Step 3: Run the Pipeline

### Basic Usage

```bash
python main.py --input path/to/your/image.jpg --target_lang Spanish
```

### Full Example

```bash
python main.py \
  --input examples/sign.jpg \
  --target_lang French \
  --output output/translated_sign.jpg \
  --save-intermediates
```

### Command-Line Arguments

- `--input` (required): Path to input image file
- `--target_lang` (required): Target language (e.g., "Spanish", "French", "German", "Japanese")
- `--output` (optional): Output path (default: `output/{input_name}_translated.jpg`)
- `--save-intermediates` (default: True): Save intermediate artifacts for debugging
- `--no-save-intermediates`: Skip saving intermediate files
- `--use-sdxl` (optional): Use SDXL inpainting instead of SD 1.5 (experimental)
- `--output-dir` (optional): Directory for outputs (default: `output`)

### Example Commands

```bash
# Basic translation to Spanish
python main.py --input photo.jpg --target_lang Spanish

# Translate to French with custom output
python main.py --input sign.png --target_lang French --output french_sign.jpg

# Translate to German without saving intermediates
python main.py --input doc.jpg --target_lang German --no-save-intermediates

# Experimental: Use SDXL inpainting for comparison
python main.py --input image.jpg --target_lang Spanish --use-sdxl
```

## Step 4: Check Results

After running, you'll find:

- **Final translated image**: `output/{filename}_translated.jpg`
- **Intermediate artifacts** (if `--save-intermediates` is enabled):
  - `ocr_visualization.jpg` - Shows detected text regions
  - `text_mask.png` - Binary mask of text regions
  - `inpainted_background.jpg` - Image after text removal
  - `ocr_metadata.json` - Structured OCR results

## Troubleshooting

### CUDA/GPU Issues

If you don't have a GPU or want to force CPU:

The code auto-detects CUDA. If you have issues, the inpainting will run on CPU (slower but works).

### Out of Memory

If you get CUDA out of memory errors:

1. The pipeline automatically normalizes large images (>2048px)
2. You can manually resize your input image before processing
3. Reduce `num_inference_steps` in `inpainting_module.py` (default: 20)

### PaddleOCR Download Issues

PaddleOCR will download models on first run. If it fails:

```bash
# Manually download PaddleOCR models
python -c "from paddleocr import PaddleOCR; PaddleOCR()"
```

### OpenAI API Errors

- Check your API key in `.env`
- Ensure you have API credits
- Check rate limits if processing many images

## Expected Processing Time

- **Small images** (< 1MP): ~30-60 seconds
- **Medium images** (1-4MP): ~1-3 minutes
- **Large images** (> 4MP): ~3-10 minutes

Times depend on:
- GPU availability (10-50x faster with CUDA)
- Number of text regions
- Image complexity

## Next Steps

- Experiment with different target languages
- Compare SD 1.5 vs SDXL results using `--use-sdxl`
- Review intermediate artifacts to understand the pipeline
- Adjust inpainting parameters in `inpainting_module.py` for different results

