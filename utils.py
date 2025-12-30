"""
Utility functions for the image translation pipeline.
Provides helper functions for image processing, file I/O, and visualization.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def ensure_output_dir(output_path: str) -> str:
    """
    Ensure the output directory exists, create if it doesn't.
    
    Args:
        output_path: Path to output file
        
    Returns:
        Directory path where output will be saved
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir if output_dir else "."


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data structure to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to output JSON file
    """
    ensure_output_dir(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data structure from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_ocr_results(image: np.ndarray, ocr_results: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and text on image for OCR visualization.
    
    Args:
        image: Input image as numpy array
        ocr_results: List of OCR results with 'bbox' and 'text' keys
        
    Returns:
        Image with bounding boxes and text annotations
    """
    vis_image = image.copy()
    
    for result in ocr_results:
        bbox = result['bbox']
        text = result['text']
        
        # Convert bbox to integer coordinates
        points = np.array(bbox, dtype=np.int32)
        
        # Draw bounding box
        cv2.polylines(vis_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw text label above bounding box
        if len(points) > 0:
            top_left = tuple(points[0])
            cv2.putText(vis_image, text[:30], (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_image


def expand_mask(mask: np.ndarray, expansion_pixels: int = 3) -> np.ndarray:
    """
    Expand binary mask by specified number of pixels to cover anti-aliased edges.
    
    Args:
        mask: Binary mask (0 or 255)
        expansion_pixels: Number of pixels to expand
        
    Returns:
        Expanded binary mask
    """
    kernel = np.ones((expansion_pixels * 2 + 1, expansion_pixels * 2 + 1), np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return expanded


def calculate_text_expansion_ratio(original: str, translated: str) -> float:
    """
    Calculate the expansion ratio of translated text compared to original.
    
    Args:
        original: Original text
        translated: Translated text
        
    Returns:
        Expansion ratio (translated_length / original_length)
    """
    if len(original) == 0:
        return 1.0
    return len(translated) / len(original)


def estimate_font_color(image: np.ndarray, bbox: List[Tuple[int, int]]) -> Tuple[int, int, int]:
    """
    Estimate font color by sampling pixels within bounding box.
    Uses median to avoid outliers from background.
    
    Args:
        image: Input image as numpy array (BGR format)
        bbox: Bounding box coordinates as list of tuples
        
    Returns:
        Estimated RGB color tuple
    """
    if len(bbox) < 3:
        return (0, 0, 0)  # Default to black
    
    # Create mask for bounding box region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # Sample pixels within the mask
    masked_region = image[mask > 0]
    
    if len(masked_region) == 0:
        return (0, 0, 0)
    
    # Use median to estimate text color (more robust than mean)
    median_color = np.median(masked_region, axis=0)
    
    # Convert BGR to RGB
    return (int(median_color[2]), int(median_color[1]), int(median_color[0]))


def get_bbox_center_and_size(bbox: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculate center point and size of bounding box.
    
    Args:
        bbox: Bounding box coordinates as list of tuples
        
    Returns:
        Tuple of (center_point, (width, height))
    """
    points = np.array(bbox)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
    size = (max_x - min_x, max_y - min_y)
    
    return center, size


def calculate_rotation_angle(bbox: List[Tuple[int, int]]) -> float:
    """
    Calculate rotation angle of bounding box from horizontal.
    
    Args:
        bbox: Bounding box coordinates as list of tuples
        
    Returns:
        Rotation angle in degrees
    """
    if len(bbox) < 2:
        return 0.0
    
    # Use first two points to estimate rotation
    p1 = np.array(bbox[0])
    p2 = np.array(bbox[1])
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

