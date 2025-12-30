"""
Input handling module for image loading and normalization.
Handles image loading, format conversion, and resolution normalization.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple


def load_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Load image from file path and return both OpenCV and PIL formats.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple of (opencv_image, pil_image)
        - opencv_image: Image as numpy array in BGR format
        - pil_image: PIL Image object in RGB format
    """
    # Load with PIL first to handle various formats
    pil_image = Image.open(image_path)
    
    # Convert to RGB if necessary (handles RGBA, L, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert PIL to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return opencv_image, pil_image


def normalize_image(image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """
    Normalize image resolution while maintaining aspect ratio.
    Useful for processing very large images efficiently.
    
    Args:
        image: Input image as numpy array
        max_dimension: Maximum dimension (width or height) to scale to
        
    Returns:
        Normalized image (may be same as input if already small enough)
    """
    height, width = image.shape[:2]
    max_current = max(height, width)
    
    if max_current <= max_dimension:
        return image
    
    # Calculate scaling factor
    scale = max_dimension / max_current
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    normalized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return normalized


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with image metadata (height, width, channels)
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    return {
        'height': height,
        'width': width,
        'channels': channels,
        'dtype': str(image.dtype)
    }

