"""
Inpainting module using Stable Diffusion for background reconstruction.
ONLY used to reconstruct background pixels where text was removed.
NEVER used to generate text.

Architectural Constraints:
- Default: Stable Diffusion 1.5 Inpainting (runwayml/stable-diffusion-inpainting)
  - More conservative, less prone to hallucination
  - Better for small, localized background reconstruction
- Optional: SDXL Inpainting (experimental comparison mode)
- Strict mask-bound inpainting
- Prompts must explicitly forbid text, letters, numbers, symbols
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from typing import Tuple, Literal
import cv2
import logging

from logger_config import get_logger

logger = get_logger(__name__)

# SDXL support (optional, experimental)
try:
    from diffusers import StableDiffusionXLInpaintPipeline
    SDXL_AVAILABLE = True
except ImportError:
    SDXL_AVAILABLE = False
    logger.warning("SDXL inpainting not available. Install diffusers>=0.21.0 for SDXL support.")


class InpaintingModule:
    """
    Stable Diffusion inpainting module for background reconstruction.
    Uses conservative prompts that explicitly forbid text generation.
    
    Default model: SD 1.5 Inpainting (runwayml/stable-diffusion-inpainting)
    - More conservative and less prone to hallucination
    - Better for small, localized background reconstruction
    
    Optional: SDXL Inpainting (experimental comparison mode)
    """
    
    # Default model - SD 1.5 Inpainting (conservative, reliable)
    DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"
    
    # SDXL model (experimental, optional)
    SDXL_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    
    def __init__(self, model_name: str = DEFAULT_MODEL, 
                 device: str = None,
                 use_sdxl: bool = False):
        """
        Initialize Stable Diffusion inpainting pipeline.
        
        Args:
            model_name: HuggingFace model name for inpainting
                       Default: "runwayml/stable-diffusion-inpainting" (SD 1.5)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_sdxl: If True, use SDXL inpainting (experimental). 
                     Overrides model_name if True.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.use_sdxl = use_sdxl
        
        # Enforce SD 1.5 as default unless explicitly requesting SDXL
        if use_sdxl:
            if not SDXL_AVAILABLE:
                raise ValueError("SDXL inpainting requested but not available. Install diffusers>=0.21.0")
            model_name = self.SDXL_MODEL
            logger.warning(f"Using SDXL inpainting (EXPERIMENTAL): {model_name}")
        else:
            # Ensure default is SD 1.5
            if model_name != self.DEFAULT_MODEL and "sdxl" in model_name.lower():
                logger.warning(f"SDXL model specified but use_sdxl=False. Using default SD 1.5 instead.")
                model_name = self.DEFAULT_MODEL
        
        self.model_name = model_name
        
        logger.info(f"Loading inpainting model: {model_name} on {device}...")
        logger.info(f"Model type: {'SDXL (experimental)' if use_sdxl else 'SD 1.5 (default, conservative)'}")
        
        # Load appropriate pipeline
        if use_sdxl and SDXL_AVAILABLE:
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None
            )
        else:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        self.pipeline = self.pipeline.to(device)
        
        # Optimize for inference
        if device == "cuda":
            self.pipeline.enable_attention_slicing()
    
    def inpaint_background(self, image: np.ndarray, mask: np.ndarray,
                          num_inference_steps: int = 20,
                          guidance_scale: float = 7.5) -> np.ndarray:
        """
        Inpaint background regions where text was removed.
        Uses conservative prompts that explicitly forbid text generation.
        
        Args:
            image: Input image as numpy array (BGR format)
            mask: Binary mask (255 = inpaint, 0 = keep)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for prompt adherence
            
        Returns:
            Inpainted image as numpy array (BGR format)
        """
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert mask to PIL (should be grayscale)
        if len(mask.shape) == 2:
            pil_mask = Image.fromarray(mask, mode='L')
        else:
            pil_mask = Image.fromarray(mask[:, :, 0], mode='L')
        
        # CRITICAL: Conservative prompt that explicitly forbids text generation
        # Stable Diffusion must NEVER generate text - only reconstruct background
        prompt = "clean background, smooth texture, no text, no letters, no numbers, no symbols, no writing, seamless continuation"
        negative_prompt = "text, letters, numbers, symbols, writing, words, characters, typography, font, alphabet, digits"
        
        logger.debug(f"Inpainting prompt: {prompt}")
        logger.debug(f"Negative prompt: {negative_prompt}")
        logger.debug(f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}")
        
        # Run inpainting
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1.0  # Full strength for complete replacement
        )
        
        # Convert result back to numpy array
        inpainted_pil = result.images[0]
        inpainted_rgb = np.array(inpainted_pil)
        
        # Convert RGB to BGR for consistency
        inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
        
        return inpainted_bgr
    
    def inpaint_with_custom_prompt(self, image: np.ndarray, mask: np.ndarray,
                                   prompt: str, negative_prompt: str = None,
                                   num_inference_steps: int = 20,
                                   guidance_scale: float = 7.5) -> np.ndarray:
        """
        Inpaint with custom prompt (for advanced use cases).
        
        Args:
            image: Input image as numpy array (BGR format)
            mask: Binary mask (255 = inpaint, 0 = keep)
            prompt: Custom prompt describing desired background
            negative_prompt: Negative prompt (default: forbids text)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for prompt adherence
            
        Returns:
            Inpainted image as numpy array (BGR format)
        """
        if negative_prompt is None:
            # CRITICAL: Default negative prompt must forbid text generation
            negative_prompt = "text, letters, numbers, symbols, writing, words, characters, typography, font, alphabet, digits"
        
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert mask to PIL
        if len(mask.shape) == 2:
            pil_mask = Image.fromarray(mask, mode='L')
        else:
            pil_mask = Image.fromarray(mask[:, :, 0], mode='L')
        
        # Run inpainting
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1.0
        )
        
        # Convert result back to numpy array
        inpainted_pil = result.images[0]
        inpainted_rgb = np.array(inpainted_pil)
        inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
        
        return inpainted_bgr

