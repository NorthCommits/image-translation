"""
OCR module for text detection and extraction.
Uses PaddleOCR for text detection with bounding boxes, orientation, and grouping.
"""

import numpy as np
from typing import List, Dict, Tuple
from paddleocr import PaddleOCR
import cv2
import logging
import os

from logger_config import get_logger

logger = get_logger(__name__)


class OCRModule:
    """
    OCR module using PaddleOCR for text detection and recognition.
    Provides structured output with bounding boxes, text content, and orientation.
    """
    
    def __init__(self, lang: str = 'en', use_angle_cls: bool = True):
        """
        Initialize PaddleOCR engine.
        
        Args:
            lang: Language code for OCR (default: 'en')
            use_angle_cls: Whether to use angle classification for rotated text
        """
        # Suppress PaddleOCR verbose output by setting environment variable
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        # Initialize PaddleOCR (show_log parameter removed in newer versions)
        logger.debug(f"Initializing PaddleOCR with lang={lang}, use_angle_cls={use_angle_cls}")
        self.ocr_engine = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        self.lang = lang
        logger.debug("PaddleOCR initialized successfully")
    
    def detect_text(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and extract text from image with bounding boxes and metadata.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries, each containing:
            - 'text': Extracted text content
            - 'bbox': Bounding box coordinates as list of tuples [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            - 'confidence': OCR confidence score
            - 'orientation': Estimated text orientation angle in degrees
        """
        # PaddleOCR expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        # Note: cls parameter removed in newer PaddleOCR versions
        # Angle classification is controlled by use_angle_cls in initialization
        logger.debug(f"Running OCR on image of size {rgb_image.shape}")
        ocr_results = self.ocr_engine.ocr(rgb_image)
        logger.debug(f"OCR completed, processing results...")
        
        # Parse results into structured format
        structured_results = []
        
        if ocr_results and len(ocr_results) > 0 and ocr_results[0] is not None:
            for line in ocr_results[0]:
                if len(line) >= 2:
                    bbox_info = line[0]  # Bounding box coordinates
                    text_info = line[1]  # (text, confidence)
                    
                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # Convert bbox to list of tuples
                        bbox = [(int(point[0]), int(point[1])) for point in bbox_info]
                        
                        # Calculate orientation from bounding box
                        orientation = self._calculate_bbox_orientation(bbox)
                        
                        structured_results.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': float(confidence),
                            'orientation': orientation
                        })
        
        return structured_results
    
    def _calculate_bbox_orientation(self, bbox: List[Tuple[int, int]]) -> float:
        """
        Calculate text orientation angle from bounding box.
        
        Args:
            bbox: Bounding box coordinates as list of tuples
            
        Returns:
            Orientation angle in degrees (0 = horizontal)
        """
        if len(bbox) < 2:
            return 0.0
        
        # Use first edge to determine orientation
        p1 = np.array(bbox[0])
        p2 = np.array(bbox[1])
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to -90 to 90 degrees
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        
        return angle_deg
    
    def group_text_by_lines(self, ocr_results: List[Dict], line_threshold: float = 0.5) -> List[List[Dict]]:
        """
        Group OCR results into lines based on vertical proximity.
        
        Args:
            ocr_results: List of OCR result dictionaries
            line_threshold: Maximum vertical distance (as fraction of average text height) to consider same line
            
        Returns:
            List of groups, where each group is a list of OCR results on the same line
        """
        if not ocr_results:
            return []
        
        # Calculate average text height
        heights = []
        for result in ocr_results:
            bbox = result['bbox']
            if len(bbox) >= 2:
                y_coords = [point[1] for point in bbox]
                height = max(y_coords) - min(y_coords)
                heights.append(height)
        
        avg_height = np.mean(heights) if heights else 50
        threshold = avg_height * line_threshold
        
        # Sort by top Y coordinate
        sorted_results = sorted(ocr_results, key=lambda r: min(p[1] for p in r['bbox']))
        
        # Group by vertical proximity
        groups = []
        current_group = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            current_top = min(p[1] for p in current_group[-1]['bbox'])
            result_top = min(p[1] for p in result['bbox'])
            
            if abs(result_top - current_top) <= threshold:
                current_group.append(result)
            else:
                groups.append(current_group)
                current_group = [result]
        
        if current_group:
            groups.append(current_group)
        
        return groups

