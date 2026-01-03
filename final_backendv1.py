import os
import cv2
import math
import easyocr
import numpy as np
import warnings
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Suppress library warnings for cleaner output
warnings.filterwarnings("ignore")
load_dotenv()

# --- CONFIGURATION ---
# Path to a reliable font. Adjust if on Linux/Windows.
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"


class ProductionTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        print("[Init] Loading OCR Engine...")
        # 'gpu=False' ensures stability on M1/M2/M3 Macs. Set True for NVIDIA PCs.
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # --- CORE UTILITIES ---
    def get_contrast_color(self, img_bgr, box):
        """
        Analyzes the background brightness behind text to determine
        if the font should be Black or White.
        """
        pts = np.array(box, dtype=np.int32)
        # Create a mask for the text box area
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Calculate average brightness of the background pixels
        mean_color = cv2.mean(img_bgr, mask=mask)[:3]
        brightness = (mean_color[0] * 0.114 + mean_color[1] * 0.587 + mean_color[2] * 0.299)

        # Return White for dark backgrounds, Black for light ones
        return (255, 255, 255) if brightness < 140 else (0, 0, 0)

    def robust_translate(self, text_list, target_lang):
        """
        Uses a system prompt specifically designed to prevent LLM refusals
        and force strict line-by-line formatting.
        """
        if not text_list: return []

        # De-duplicate while keeping order (critical for diagrams with repeated labels)
        unique_texts = list(dict.fromkeys(text_list))

        print(f"   [AI] Translating {len(unique_texts)} unique segments...")

        system_msg = (
            "You are a professional GUI localization tool. "
            f"Translate the following UI strings to {target_lang}. "
            "Do not explain. Do not refuse. "
            "Return EXACTLY one translation line per input line. "
            "Maintain capitalization style (Title Case vs Sentence case)."
        )

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": "\n".join(unique_texts)}
                ],
                temperature=0.0
            )
            translated_lines = resp.choices[0].message.content.strip().split('\n')

            # Create a lookup map to restore order
            trans_map = {src: trans for src, trans in zip(unique_texts, translated_lines)}
            return [trans_map.get(t, t) for t in text_list]

        except Exception as e:
            print(f"   [Error] Translation API failed: {e}")
            return text_list  # Fallback to original

    def calculate_optimal_font_size(self, draw, text, w_box, h_box, font_path, min_size=8, max_size=100):
        """
        Calculate optimal font size that fits within the box dimensions.
        More conservative approach to match original text proportions.
        """
        # Start with a conservative estimate based on box height
        fontsize = int(h_box * 0.65)  # Reduced from 0.9 for better proportions
        fontsize = min(fontsize, max_size)  # Cap at max size
        
        try:
            font = ImageFont.truetype(font_path, fontsize)
        except:
            font = ImageFont.load_default()
            return font, fontsize
        
        # Measure text and shrink if needed
        while fontsize > min_size:
            # Get text dimensions using getbbox for accuracy
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits with some padding (1.1 = 10% padding)
            if text_width <= w_box * 0.95 and text_height <= h_box * 0.95:
                break
                
            fontsize -= 1
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except:
                pass
        
        return font, fontsize

    # --- LINEAR ENGINE (For Banners) ---
    def process_linear(self, img, original_alpha, results, target_lang):
        """ 
        Optimized for banners, headers, and document strips.
        Improved font sizing for better proportions.
        """
        print("   [Mode] Linear Layout Detected")

        # 1. Filter Valid Text
        valid_items = [r for r in results if r[2] > 0.4]
        texts = [r[1] for r in valid_items]
        boxes = [r[0] for r in valid_items]

        if not texts:
            # No text found, return original with alpha
            if original_alpha is not None:
                b, g, r = cv2.split(img)
                final_bgra = cv2.merge([b, g, r, original_alpha])
                return Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        translated = self.robust_translate(texts, target_lang)

        # 2. Clean Text Areas (Inpainting)
        clean_img = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Dilate mask to ensure full coverage of old letters
        mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)
        clean_img = cv2.inpaint(clean_img, mask, 3, cv2.INPAINT_TELEA)

        # 3. Preserve Original Alpha Channel
        if original_alpha is not None:
            b, g, r = cv2.split(clean_img)
            final_bgra = cv2.merge([b, g, r, original_alpha])
            pil_img = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
            print("   [Alpha] Preserved original transparency")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_img)

        # 4. Render with improved font sizing
        for i, (box, text) in enumerate(zip(boxes, translated)):
            text = text.replace('"', '').strip()
            pts = np.array(box, dtype=np.int32)

            # Box Dimensions
            w_box = np.linalg.norm(pts[1] - pts[0])
            h_box = np.linalg.norm(pts[3] - pts[0])
            x, y = pts[0][0], pts[0][1]

            # Determine Color
            color = self.get_contrast_color(img, box)

            # Calculate optimal font size
            font, fontsize = self.calculate_optimal_font_size(
                draw, text, w_box, h_box, FONT_PATH
            )

            # Draw text
            draw.text((x, y), text, font=font, fill=color)

        return pil_img

    # --- RADIAL ENGINE IMPROVED (For Diagrams) ---
    def process_radial(self, img, original_alpha, results, target_lang):
        """ 
        Improved version for Circular Charts.
        Better positioning, reduced overlap, smarter rotation.
        """
        print("   [Mode] Radial Layout Detected (Enhanced)")
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Helper: Convert Box to Polar (Radius, Angle)
        def get_polar_stats(box):
            pts = np.array(box, dtype=np.int32)
            cx, cy = np.mean(pts, axis=0)
            dx, dy = cx - center[0], cy - center[1]
            radius = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.degrees(math.atan2(dy, dx))
            # Normalize angle to 0-360
            if angle < 0: angle += 360
            return radius, angle, (cx, cy), pts

        # 1. Pre-process and filter items
        valid_items = [r for r in results if r[2] > 0.5]  # Higher confidence threshold
        
        items = []
        for (box, text, prob) in valid_items:
            r, theta, centroid, pts = get_polar_stats(box)
            
            # Calculate box dimensions for font sizing
            w_box = np.linalg.norm(pts[1] - pts[0])
            h_box = np.linalg.norm(pts[3] - pts[0])
            
            items.append({
                'box': box, 
                'text': text,
                'r': r, 
                'theta': theta, 
                'center': centroid,
                'w_box': w_box,
                'h_box': h_box
            })

        if not items:
            # No text found, return original with alpha
            if original_alpha is not None:
                b, g, r = cv2.split(img)
                final_bgra = cv2.merge([b, g, r, original_alpha])
                return Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 2. Smart grouping by radial zones (inner, middle, outer rings)
        # Categorize by radius
        max_radius = max([item['r'] for item in items])
        
        for item in items:
            if item['r'] < max_radius * 0.33:
                item['zone'] = 'inner'
            elif item['r'] < max_radius * 0.66:
                item['zone'] = 'middle'
            else:
                item['zone'] = 'outer'

        # 3. Translate all texts individually
        texts = [item['text'] for item in items]
        translated = self.robust_translate(texts, target_lang)

        # 4. Very Aggressive Inpainting
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for item in items:
            pts = np.array(item['box'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Extra aggressive dilation based on zone
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=4)
        clean_img = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)

        # 5. Preserve Original Alpha Channel
        if original_alpha is not None:
            b, g, r = cv2.split(clean_img)
            final_bgra = cv2.merge([b, g, r, original_alpha])
            pil_img = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
            print("   [Alpha] Preserved original transparency")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))

        # 6. Render each text with smart rotation and sizing
        print(f"   [Render] Drawing {len(items)} text regions with smart rotation...")
        
        for i, item in enumerate(items):
            if i >= len(translated): break
            
            text = translated[i].replace('"', '').strip()
            if not text: continue
            
            cx, cy = item['center']
            theta = item['theta']
            w_box = item['w_box']
            h_box = item['h_box']
            zone = item['zone']
            
            # Determine text color based on background
            color = self.get_contrast_color(img, item['box'])
            
            # Calculate rotation angle
            rotation_angle = theta + 90
            
            # Adjust rotation to keep text readable
            if 90 < theta < 270:
                rotation_angle = theta - 90
            
            # Zone-specific font sizing (outer text smaller, inner text larger)
            if zone == 'outer':
                max_font = 16
                height_ratio = 0.6
            elif zone == 'middle':
                max_font = 18
                height_ratio = 0.7
            else:  # inner
                max_font = 20
                height_ratio = 0.7
            
            # Calculate appropriate font size
            fontsize = int(min(h_box * height_ratio, max_font))
            fontsize = max(fontsize, 8)  # Minimum 8pt
            
            try:
                font = ImageFont.truetype(FONT_PATH, fontsize)
            except:
                font = ImageFont.load_default()
            
            # Shrink font to fit width if needed (more conservative)
            temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            while fontsize > 8:
                bbox = temp_draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                
                if text_w <= w_box * 0.9: break  # 90% of box width
                fontsize -= 1
                try:
                    font = ImageFont.truetype(FONT_PATH, fontsize)
                except:
                    pass
            
            # Create temporary image for rotated text
            temp_size = int(max(w_box, h_box) * 2.5)
            temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Draw text in center of temp image
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = (temp_size - text_w) / 2
            text_y = (temp_size - text_h) / 2
            temp_draw.text((text_x, text_y), text, font=font, fill=color)
            
            # Rotate the temporary image
            rotated = temp_img.rotate(-rotation_angle, expand=False, resample=Image.BICUBIC)
            
            # Calculate paste position
            paste_x = int(cx - temp_size / 2)
            paste_y = int(cy - temp_size / 2)
            
            # Paste rotated text onto main image
            pil_img.paste(rotated, (paste_x, paste_y), rotated)

        return pil_img

    # --- MAIN PIPELINE ---
    def process(self, image_path, target_lang="Spanish"):
        if not os.path.exists(image_path): 
            print("File not found."); 
            return
        
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")

        # 1. Load Image WITH Alpha Channel Support
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None: 
            print("Error reading image."); 
            return

        # 2. Extract and Store Alpha Channel
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:  # Image has alpha channel
            print("   [Detected] Image has transparency")
            original_alpha = img[:, :, 3]  # Extract alpha
            img = img[:, :, :3]  # Work with BGR only
        elif len(img.shape) == 3 and img.shape[2] == 3:
            print("   [Detected] Image has no transparency")
        
        # Convert to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 3. Detect Text
        print("   [OCR] Detecting text...")
        results = self.reader.readtext(img)
        print(f"   [OCR] Found {len(results)} text regions")

        # 4. Layout Detection
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Logic: Diagrams are usually square-ish (Ratio < 1.4). Banners are wide.
        if aspect_ratio > 1.4:
            result_img = self.process_linear(img, original_alpha, results, target_lang)
        else:
            # Double check for circles
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
            if circles is not None or aspect_ratio < 1.2:
                result_img = self.process_radial(img, original_alpha, results, target_lang)
            else:
                result_img = self.process_linear(img, original_alpha, results, target_lang)

        # 5. Save with Transparency Support
        out_name = f"translated_{os.path.basename(image_path)}"
        out_name = os.path.splitext(out_name)[0] + ".png"  # Force PNG
        result_img.save(out_name, "PNG")
        print(f"SUCCESS! Saved to: {out_name}")


if __name__ == "__main__":
    t = ProductionTranslator()
    while True:
        p = input("Image Path (or 'q'): ").strip()
        if p.lower() == 'q': break
        l = input("Target Language: ").strip()
        t.process(p, l)