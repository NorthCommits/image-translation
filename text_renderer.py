"""
Text rendering module for deterministic text placement.
Renders translated text onto images using graphics libraries (NOT diffusion).
Matches original placement, rotation, and approximate style.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Tuple
import math


class TextRenderer:
    """
    Deterministic text renderer for placing translated text.
    Estimates font properties from original image and renders text programmatically.
    """
    
    def __init__(self):
        """
        Initialize text renderer.
        """
        self.default_font_path = None
        self._load_default_font()
    
    def _load_default_font(self):
        """
        Load default font for text rendering.
        Tries to find a suitable system font.
        """
        try:
            # Try to load a common system font
            self.default_font_path = "/System/Library/Fonts/Helvetica.ttc"  # macOS
            ImageFont.truetype(self.default_font_path, size=20)
        except:
            try:
                self.default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
                ImageFont.truetype(self.default_font_path, size=20)
            except:
                # Fall back to default font
                self.default_font_path = None
    
    def render_text_on_image(self, image: np.ndarray, 
                            translated_results: List[Dict]) -> np.ndarray:
        """
        Render translated text onto image at original bounding box locations.
        
        Args:
            image: Input image as numpy array (BGR format)
            translated_results: List of translated OCR results with 'bbox', 'translated_text', etc.
            
        Returns:
            Image with rendered text as numpy array (BGR format)
        """
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to RGBA if we need alpha compositing for rotated text
        needs_alpha = any(abs(r.get('orientation', 0.0)) > 1.0 for r in translated_results)
        if needs_alpha and pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # Create drawing context
        draw = ImageDraw.Draw(pil_image)
        
        for result in translated_results:
            bbox = result['bbox']
            translated_text = result.get('translated_text', result.get('text', ''))
            orientation = result.get('orientation', 0.0)
            expansion_ratio = result.get('expansion_ratio', 1.0)
            
            if not translated_text:
                continue
            
            # Estimate font properties from original image region
            font_size, font_color = self._estimate_font_properties(image, bbox, translated_text, expansion_ratio)
            
            # Render text (may update pil_image for rotated text)
            pil_image = self._render_text_at_bbox(draw, pil_image, bbox, translated_text, font_size, font_color, orientation)
            
            # Recreate draw object if image was updated
            if pil_image.mode == 'RGBA' or needs_alpha:
                draw = ImageDraw.Draw(pil_image)
        
        # Convert back to RGB if needed (from RGBA)
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB by compositing on white background
            rgb_background = Image.new('RGB', pil_image.size, (255, 255, 255))
            pil_image = Image.alpha_composite(rgb_background.convert('RGBA'), pil_image).convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert back to BGR numpy array
        result_rgb = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result_bgr
    
    def _estimate_font_properties(self, image: np.ndarray, bbox: List[Tuple[int, int]],
                                 text: str, expansion_ratio: float) -> Tuple[int, Tuple[int, int, int]]:
        """
        Estimate font size and color from original image region.
        
        Args:
            image: Input image as numpy array
            bbox: Bounding box coordinates
            text: Text to render
            expansion_ratio: Text expansion ratio after translation
            
        Returns:
            Tuple of (font_size, font_color_rgb)
        """
        # Calculate bounding box dimensions
        points = np.array(bbox)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        # Estimate font size based on bounding box height
        # Adjust for text expansion
        base_font_size = int(height * 0.7)
        adjusted_font_size = int(base_font_size / max(expansion_ratio, 1.0))
        
        # Ensure minimum and maximum font sizes
        font_size = max(12, min(adjusted_font_size, 200))
        
        # Estimate font color by sampling pixels in bounding box
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        
        # Sample pixels within the mask
        masked_region = image[mask > 0]
        
        if len(masked_region) > 0:
            # Use median to estimate text color (more robust than mean)
            median_color = np.median(masked_region, axis=0)
            font_color = (int(median_color[2]), int(median_color[1]), int(median_color[0]))  # BGR to RGB
        else:
            font_color = (0, 0, 0)  # Default to black
        
        return font_size, font_color
    
    def _render_text_at_bbox(self, draw: ImageDraw.Draw, pil_image: Image.Image,
                            bbox: List[Tuple[int, int]], text: str, font_size: int,
                            font_color: Tuple[int, int, int], orientation: float) -> Image.Image:
        """
        Render text at bounding box location with specified properties.
        
        Args:
            draw: PIL ImageDraw object
            pil_image: PIL Image object for compositing rotated text
            bbox: Bounding box coordinates
            text: Text to render
            font_size: Font size in pixels
            font_color: Font color as RGB tuple
            orientation: Text rotation angle in degrees
            
        Returns:
            Updated PIL Image (may be new object if compositing was used)
        """
        if not text:
            return pil_image
        
        # Load font
        try:
            if self.default_font_path:
                font = ImageFont.truetype(self.default_font_path, size=font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Calculate bounding box center
        points = np.array(bbox)
        center_x = int(points[:, 0].mean())
        center_y = int(points[:, 1].mean())
        
        # Get text bounding box
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Calculate position (centered in original bbox)
        x = center_x - text_width // 2
        y = center_y - text_height // 2
        
        # Handle rotation
        if abs(orientation) > 1.0:  # Only rotate if significant
            # Create temporary image for rotated text
            temp_image = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_image)
            temp_draw.text((10, 10), text, font=font, fill=font_color)
            
            # Rotate temporary image
            rotated_temp = temp_image.rotate(-orientation, expand=True, fillcolor=(0, 0, 0, 0))
            
            # Paste rotated text onto main image
            # Calculate paste position to center rotated text
            rot_width, rot_height = rotated_temp.size
            paste_x = center_x - rot_width // 2
            paste_y = center_y - rot_height // 2
            
            # Use alpha composite for proper blending
            if rotated_temp.mode == 'RGBA':
                # Ensure main image is RGBA
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')
                
                # Create overlay image with same size as main image
                overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                overlay.paste(rotated_temp, (paste_x, paste_y), rotated_temp)
                
                # Composite overlay onto main image
                pil_image = Image.alpha_composite(pil_image, overlay)
            else:
                pil_image.paste(rotated_temp, (paste_x, paste_y))
        else:
            # No rotation, simple text rendering
            draw.text((x, y), text, font=font, fill=font_color)
        
        return pil_image

