"""
Main pipeline orchestrator for image translation.
Coordinates all stages: OCR, translation, inpainting, and text rendering.
"""

import os
import numpy as np
import cv2
from typing import Dict, Optional
import json
import logging

from input_handler import load_image, normalize_image, get_image_info
from ocr_module import OCRModule
from translation_module import TranslationModule
from mask_generator import MaskGenerator
from inpainting_module import InpaintingModule
from text_renderer import TextRenderer
from utils import save_json, visualize_ocr_results, ensure_output_dir
from logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class ImageTranslationPipeline:
    """
    Main pipeline for translating text in images.
    Orchestrates all processing stages sequentially.
    """
    
    def __init__(self, target_language: str, 
                 inpainting_model: str = "runwayml/stable-diffusion-inpainting",
                 use_sdxl: bool = False,
                 save_intermediates: bool = True,
                 output_dir: str = "output"):
        """
        Initialize pipeline with configuration.
        
        Args:
            target_language: Target language for translation
            inpainting_model: Stable Diffusion model name (default: SD 1.5)
            use_sdxl: If True, use SDXL inpainting (experimental comparison mode)
                     Default: False (uses SD 1.5, more conservative)
            save_intermediates: Whether to save intermediate artifacts
            output_dir: Directory for saving outputs
        """
        self.target_language = target_language
        self.save_intermediates = save_intermediates
        self.output_dir = output_dir
        self.use_sdxl = use_sdxl
        
        # Initialize modules
        logger.info("Initializing OCR module...")
        self.ocr_module = OCRModule(lang='en', use_angle_cls=True)
        logger.debug("OCR module initialized")
        
        logger.info("Initializing translation module...")
        self.translation_module = TranslationModule()
        logger.debug("Translation module initialized")
        
        logger.info("Initializing mask generator...")
        self.mask_generator = MaskGenerator(expansion_pixels=3)
        logger.debug("Mask generator initialized")
        
        logger.info("Initializing inpainting module...")
        self.inpainting_module = InpaintingModule(
            model_name=inpainting_model,
            use_sdxl=use_sdxl
        )
        logger.debug("Inpainting module initialized")
        
        logger.info("Initializing text renderer...")
        self.text_renderer = TextRenderer()
        logger.debug("Text renderer initialized")
        
        logger.info("Pipeline initialized successfully")
    
    def process(self, input_image_path: str, output_image_path: str) -> Dict:
        """
        Process image through complete translation pipeline.
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to save output image
            
        Returns:
            Dictionary with processing results and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting Image Translation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Input: {input_image_path}")
        logger.info(f"Target language: {self.target_language}")
        logger.info(f"Output: {output_image_path}")
        logger.info(f"Inpainting model: {self.inpainting_module.model_name}")
        logger.info(f"Model type: {'SDXL (experimental)' if self.use_sdxl else 'SD 1.5 (default)'}")
        
        # Stage 1: Input Handling
        logger.info("Stage 1: Loading and normalizing image...")
        opencv_image, pil_image = load_image(input_image_path)
        image_info = get_image_info(opencv_image)
        logger.info(f"Image size: {image_info['width']}x{image_info['height']} ({image_info['channels']} channels)")
        
        # Normalize if too large
        if max(image_info['width'], image_info['height']) > 2048:
            logger.info("Normalizing image size (max dimension > 2048px)...")
            opencv_image = normalize_image(opencv_image, max_dimension=2048)
            image_info = get_image_info(opencv_image)
            logger.info(f"Normalized size: {image_info['width']}x{image_info['height']}")
        
        # Stage 2: OCR
        logger.info("Stage 2: Detecting and extracting text (OCR)...")
        ocr_results = self.ocr_module.detect_text(opencv_image)
        logger.info(f"Detected {len(ocr_results)} text regions")
        
        if len(ocr_results) == 0:
            logger.warning("No text detected. Returning original image.")
            cv2.imwrite(output_image_path, opencv_image)
            return {
                'status': 'no_text_detected',
                'ocr_results': [],
                'translated_results': []
            }
        
        # Log OCR details
        for i, result in enumerate(ocr_results[:5]):  # Log first 5 for debugging
            logger.debug(f"OCR result {i+1}: '{result['text'][:50]}...' (confidence: {result['confidence']:.2f})")
        
        # Save OCR visualization
        if self.save_intermediates:
            ocr_vis = visualize_ocr_results(opencv_image, ocr_results)
            ocr_vis_path = os.path.join(self.output_dir, "ocr_visualization.jpg")
            ensure_output_dir(ocr_vis_path)
            cv2.imwrite(ocr_vis_path, ocr_vis)
            logger.debug(f"Saved OCR visualization: {ocr_vis_path}")
            
            # Save OCR metadata
            ocr_metadata_path = os.path.join(self.output_dir, "ocr_metadata.json")
            save_json({
                'image_info': image_info,
                'ocr_results': ocr_results
            }, ocr_metadata_path)
            logger.debug(f"Saved OCR metadata: {ocr_metadata_path}")
        
        # Stage 3: Translation
        logger.info("Stage 3: Translating text...")
        translated_results = self.translation_module.translate_ocr_results(
            ocr_results, self.target_language
        )
        logger.info(f"Translated {len(translated_results)} text segments")
        
        # Log expansion ratios
        expansion_ratios = [r.get('expansion_ratio', 1.0) for r in translated_results]
        avg_expansion = np.mean(expansion_ratios) if expansion_ratios else 1.0
        max_expansion = max(expansion_ratios) if expansion_ratios else 1.0
        logger.info(f"Text expansion - Average: {avg_expansion:.2f}x, Max: {max_expansion:.2f}x")
        
        # Stage 4: Mask Creation
        logger.info("Stage 4: Creating inpainting masks...")
        mask = self.mask_generator.create_mask_from_bboxes(
            opencv_image.shape[:2], ocr_results
        )
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        mask_percentage = (mask_area / total_area) * 100
        logger.info(f"Mask covers {mask_percentage:.2f}% of image ({mask_area:,} pixels)")
        
        if self.save_intermediates:
            mask_path = os.path.join(self.output_dir, "text_mask.png")
            self.mask_generator.save_mask(mask, mask_path)
            logger.debug(f"Saved mask: {mask_path}")
        
        # Stage 5: Inpainting
        logger.info("Stage 5: Inpainting background (removing original text)...")
        logger.info("This may take a while depending on image size and GPU availability...")
        try:
            inpainted_image = self.inpainting_module.inpaint_background(
                opencv_image, mask, num_inference_steps=20, guidance_scale=7.5
            )
            logger.info("Inpainting completed successfully")
        except Exception as e:
            logger.error(f"Inpainting failed: {e}", exc_info=True)
            raise
        
        if self.save_intermediates:
            inpainted_path = os.path.join(self.output_dir, "inpainted_background.jpg")
            cv2.imwrite(inpainted_path, inpainted_image)
            logger.debug(f"Saved inpainted background: {inpainted_path}")
        
        # Stage 6: Text Rendering
        logger.info("Stage 6: Rendering translated text...")
        final_image = self.text_renderer.render_text_on_image(
            inpainted_image, translated_results
        )
        logger.info("Text rendering completed")
        
        # Stage 7: Final Output
        logger.info("Stage 7: Saving final image...")
        ensure_output_dir(output_image_path)
        cv2.imwrite(output_image_path, final_image)
        logger.info(f"Saved final image: {output_image_path}")
        
        # Prepare results summary
        results = {
            'status': 'success',
            'image_info': image_info,
            'ocr_results_count': len(ocr_results),
            'translated_results_count': len(translated_results),
            'average_expansion_ratio': float(avg_expansion),
            'mask_percentage': float(mask_percentage),
            'inpainting_model': self.inpainting_module.model_name,
            'inpainting_model_type': 'SDXL (experimental)' if self.use_sdxl else 'SD 1.5 (default, conservative)',
            'output_path': output_image_path
        }
        
        if self.save_intermediates:
            results['intermediate_artifacts'] = {
                'ocr_visualization': os.path.join(self.output_dir, "ocr_visualization.jpg"),
                'ocr_metadata': os.path.join(self.output_dir, "ocr_metadata.json"),
                'text_mask': os.path.join(self.output_dir, "text_mask.png"),
                'inpainted_background': os.path.join(self.output_dir, "inpainted_background.jpg")
            }
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Status: {results['status']}")
        logger.info(f"Text regions: {results['ocr_results_count']} detected, {results['translated_results_count']} translated")
        logger.info(f"Expansion ratio: {results['average_expansion_ratio']:.2f}x")
        logger.info(f"Mask coverage: {results['mask_percentage']:.2f}%")
        logger.info(f"Output: {results['output_path']}")
        
        return results

