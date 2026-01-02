import os
import cv2
import math
import easyocr
import textwrap
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

    # --- LINEAR ENGINE (For Banners) ---
    def process_linear(self, img, results, target_lang):
        """ Optimized for banners, headers, and document strips. """
        print("   [Mode] Linear Layout Detected")

        # 1. Filter Valid Text
        valid_items = [r for r in results if r[2] > 0.4]
        texts = [r[1] for r in valid_items]
        boxes = [r[0] for r in valid_items]

        if not texts: return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        translated = self.robust_translate(texts, target_lang)

        # 2. Advanced Background Removal (Magic Transparency)
        # Check if background is uniform (White/Black) and remove it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 230:  # White BG
            _, alpha = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
            print("   [Magic] Converting White BG to Transparent")
        elif np.mean(gray) < 30:  # Black BG
            _, alpha = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            print("   [Magic] Converting Black BG to Transparent")
        else:
            alpha = np.ones(gray.shape, dtype=np.uint8) * 255

        # 3. Clean Text Areas (Inpainting)
        clean_img = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Dilate mask to ensure full coverage of old letters
        mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)
        clean_img = cv2.inpaint(clean_img, mask, 3, cv2.INPAINT_TELEA)

        # Merge Alpha
        b, g, r = cv2.split(clean_img)
        final_bgra = cv2.merge([b, g, r, alpha])
        pil_img = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
        draw = ImageDraw.Draw(pil_img)

        # 4. Render
        for i, (box, text) in enumerate(zip(boxes, translated)):
            text = text.replace('"', '').strip()
            pts = np.array(box, dtype=np.int32)

            # Box Dimensions
            w_box = np.linalg.norm(pts[1] - pts[0])
            h_box = np.linalg.norm(pts[3] - pts[0])
            x, y = pts[0][0], pts[0][1]

            # Determine Color
            color = self.get_contrast_color(img, box)

            # Shrink-to-Fit Logic
            fontsize = int(h_box * 0.9)
            try:
                font = ImageFont.truetype(FONT_PATH, fontsize)
            except:
                font = ImageFont.load_default()

            while fontsize > 8:
                if hasattr(draw, "textlength"):
                    w = draw.textlength(text, font=font)
                else:
                    w = draw.textbbox((0, 0), text, font=font)[2]
                if w <= w_box * 1.05: break
                fontsize -= 1
                try:
                    font = ImageFont.truetype(FONT_PATH, fontsize)
                except:
                    pass

            draw.text((x, y), text, font=font, fill=color)

        return pil_img

    # --- RADIAL ENGINE (For Diagrams) ---
    def process_radial(self, img, results, target_lang):
        """ Optimized for Circular Charts. Groups text by sector. """
        print("   [Mode] Radial Layout Detected")
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
            return radius, angle, (cx, cy)

        # 1. Pre-process items
        items = []
        for (box, text, prob) in results:
            if prob < 0.4: continue
            r, theta, centroid = get_polar_stats(box)
            items.append({
                'box': box, 'text': text,
                'r': r, 'theta': theta, 'center': centroid
            })

        # 2. Smart Clustering (The "Wedge" Logic)
        groups = []
        while items:
            curr = items.pop(0)
            cluster = [curr]

            # Look for neighbors in the same wedge
            others = [x for x in items]
            for other in others:
                # Angle difference (handling 359 vs 1 degree)
                diff = abs(curr['theta'] - other['theta'])
                if diff > 180: diff = 360 - diff

                # Clustering Rules:
                # 1. Same Wedge: Angle diff < 25 degrees
                # 2. Connected Text: Radius diff < 150px
                if diff < 25 and abs(curr['r'] - other['r']) < 150:
                    cluster.append(other)
                    if other in items: items.remove(other)

            # Sort cluster by radius (reading inner-to-outer usually correct for headers)
            cluster.sort(key=lambda x: x['r'])

            # Combine text
            full_text = " ".join([x['text'] for x in cluster])

            # Calculate the visual center of the whole cluster
            all_pts = np.concatenate([np.array(x['box']) for x in cluster])
            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)

            groups.append({
                'text': full_text,
                'cluster': cluster,
                'bounds': (x_min, y_min, x_max, y_max),
                'visual_center': ((x_min + x_max) / 2, (y_min + y_max) / 2)
            })

        # 3. Translate
        texts = [g['text'] for g in groups]
        translated = self.robust_translate(texts, target_lang)

        # 4. Smart Inpainting (Preserve Gradients)
        # We only mask the WHITE text pixels, not the colored background
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for g in groups:
            for item in g['cluster']:
                pts = np.array(item['box'], dtype=np.int32)
                x, y, w_b, h_b = cv2.boundingRect(pts)

                # Padding to catch anti-aliasing
                pad = 5
                roi = img[max(0, y - pad):y + h_b + pad, max(0, x - pad):x + w_b + pad]

                # Detect Text Pixels
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Adaptive threshold for light text on dark background
                _, tmask = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)

                try:
                    mask[max(0, y - pad):y + h_b + pad, max(0, x - pad):x + w_b + pad] = tmask
                except:
                    pass

        # Dilate heavily to eat the "ghosts"
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # 5. Render Blocks
        pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for i, group in enumerate(groups):
            if i >= len(translated): break
            text = translated[i].replace('"', '').strip()

            cx, cy = group['visual_center']
            (x1, y1, x2, y2) = group['bounds']
            w_box = x2 - x1
            h_box = y2 - y1

            # Dynamic Font Sizing
            fontsize = 18
            try:
                font = ImageFont.truetype(FONT_PATH, fontsize)
            except:
                font = ImageFont.load_default()

            # Wrapping Logic
            char_w = 8
            # Approx chars that fit in the box width
            wrap_width = max(10, int(w_box / char_w))
            lines = textwrap.wrap(text, width=wrap_width)

            # Check vertical fit
            total_h = len(lines) * (fontsize + 4)
            start_y = cy - total_h / 2

            # Draw Centered
            for line in lines:
                if hasattr(draw, "textlength"):
                    lw = draw.textlength(line, font=font)
                else:
                    lw = draw.textbbox((0, 0), line, font=font)[2]

                draw.text((cx - lw / 2, start_y), line, font=font, fill=(255, 255, 255))
                start_y += fontsize + 4

        return pil_img

    # --- MAIN PIPELINE ---
    def process(self, image_path, target_lang="Spanish"):
        if not os.path.exists(image_path): print("File not found."); return
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")

        # 1. Load (Preserve Alpha if possible, but process as BGR)
        img = cv2.imread(image_path)
        if img is None: print("Error reading image."); return

        # 2. Detect Text
        results = self.reader.readtext(img)

        # 3. Layout Detection
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Logic: Diagrams are usually square-ish (Ratio < 1.4). Banners are wide.
        if aspect_ratio > 1.4:
            result_img = self.process_linear(img, results, target_lang)
        else:
            # Double check for circles
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
            if circles is not None or aspect_ratio < 1.2:
                result_img = self.process_radial(img, results, target_lang)
            else:
                result_img = self.process_linear(img, results, target_lang)

        # 4. Save
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