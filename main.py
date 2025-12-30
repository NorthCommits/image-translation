"""
Main entry point for image translation pipeline.
Command-line interface for processing images.
"""

import argparse
import os
import sys
import logging
from pipeline import ImageTranslationPipeline
from logger_config import setup_logger


def main():
    """
    Main entry point for image translation.
    Parses command-line arguments and runs the pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Translate text in images while preserving visual appearance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input image.jpg --target_lang Spanish
  python main.py --input photo.png --target_lang French --output translated.png
  python main.py --input doc.jpg --target_lang German --no-save-intermediates
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--target_lang',
        type=str,
        required=True,
        help='Target language for translation (e.g., "Spanish", "French", "German")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path for output image (default: output_translated.jpg)'
    )
    
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        default=True,
        help='Save intermediate artifacts for debugging (default: True)'
    )
    
    parser.add_argument(
        '--no-save-intermediates',
        dest='save_intermediates',
        action='store_false',
        help='Do not save intermediate artifacts'
    )
    
    parser.add_argument(
        '--inpainting-model',
        type=str,
        default='runwayml/stable-diffusion-inpainting',
        help='Stable Diffusion inpainting model name (default: runwayml/stable-diffusion-inpainting, SD 1.5)'
    )
    
    parser.add_argument(
        '--use-sdxl',
        action='store_true',
        default=False,
        help='Use SDXL inpainting (EXPERIMENTAL, for comparison). Default: False (uses SD 1.5, more conservative)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory for saving outputs and intermediates (default: output)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable logging to file (logs only to console)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logger(
        name="image_translation",
        log_level=args.log_level,
        log_dir="logs",
        log_to_file=not args.no_log_file,
        log_to_console=True
    )
    
    logger = logging.getLogger("image_translation")
    logger.info("=" * 60)
    logger.info("Image Translation Pipeline")
    logger.info("=" * 60)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(args.output_dir, f"{base_name}_translated.jpg")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run pipeline
    try:
        pipeline = ImageTranslationPipeline(
            target_language=args.target_lang,
            inpainting_model=args.inpainting_model,
            use_sdxl=args.use_sdxl,
            save_intermediates=args.save_intermediates,
            output_dir=args.output_dir
        )
        
        results = pipeline.process(args.input, args.output)
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Processing Summary")
        logger.info("=" * 60)
        logger.info(f"Status: {results['status']}")
        if results['status'] == 'success':
            logger.info(f"Text regions detected: {results['ocr_results_count']}")
            logger.info(f"Text regions translated: {results['translated_results_count']}")
            logger.info(f"Average expansion ratio: {results['average_expansion_ratio']:.2f}")
            logger.info(f"Mask coverage: {results['mask_percentage']:.2f}%")
            logger.info(f"Inpainting model: {results.get('inpainting_model_type', 'SD 1.5 (default)')}")
            logger.info(f"Output saved to: {results['output_path']}")
            
            if args.save_intermediates and 'intermediate_artifacts' in results:
                logger.info("Intermediate artifacts saved:")
                for name, path in results['intermediate_artifacts'].items():
                    logger.info(f"  - {name}: {path}")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

