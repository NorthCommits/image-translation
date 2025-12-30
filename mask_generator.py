"""
Mask generation module for creating inpainting masks from OCR bounding boxes.
Converts bounding boxes into pixel-accurate binary masks with optional expansion.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from PIL import Image


class MaskGenerator:
    """
    Generates binary masks from OCR bounding boxes for inpainting.
    Supports rotated bounding boxes and mask expansion for anti-aliasing.
    """
    
    def __init__(self, expansion_pixels: int = 3):
        """
        Initialize mask generator.
        
        Args:
            expansion_pixels: Number of pixels to expand masks (default: 3)
        """
        self.expansion_pixels = expansion_pixels
    
    def create_mask_from_bboxes(self, image_shape: Tuple[int, int], 
                                 ocr_results: List[Dict]) -> np.ndarray:
        """
        Create binary mask from OCR bounding boxes.
        
        Args:
            image_shape: Shape of image (height, width)
            ocr_results: List of OCR results with 'bbox' key
            
        Returns:
            Binary mask (0 = keep, 255 = inpaint) as numpy array
        """
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        
        for result in ocr_results:
            bbox = result['bbox']
            if len(bbox) >= 3:
                # Convert bbox to numpy array of points
                points = np.array(bbox, dtype=np.int32)
                
                # Fill polygon defined by bounding box
                cv2.fillPoly(mask, [points], 255)
        
        # Expand mask to cover anti-aliased edges
        if self.expansion_pixels > 0:
            mask = self._expand_mask(mask)
        
        return mask
    
    def _expand_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Expand binary mask by specified number of pixels.
        
        Args:
            mask: Binary mask (0 or 255)
            
        Returns:
            Expanded binary mask
        """
        kernel_size = self.expansion_pixels * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded = cv2.dilate(mask, kernel, iterations=1)
        return expanded
    
    def create_individual_masks(self, image_shape: Tuple[int, int],
                                ocr_results: List[Dict]) -> List[np.ndarray]:
        """
        Create separate mask for each OCR result.
        Useful for processing text regions individually.
        
        Args:
            image_shape: Shape of image (height, width)
            ocr_results: List of OCR results with 'bbox' key
            
        Returns:
            List of binary masks, one per OCR result
        """
        individual_masks = []
        
        for result in ocr_results:
            mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
            bbox = result['bbox']
            
            if len(bbox) >= 3:
                points = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
                
                if self.expansion_pixels > 0:
                    mask = self._expand_mask(mask)
            
            individual_masks.append(mask)
        
        return individual_masks
    
    def save_mask(self, mask: np.ndarray, filepath: str) -> None:
        """
        Save mask as image file for debugging.
        
        Args:
            mask: Binary mask to save
            filepath: Path to save mask image
        """
        # Convert to PIL Image and save
        mask_image = Image.fromarray(mask)
        mask_image.save(filepath)

