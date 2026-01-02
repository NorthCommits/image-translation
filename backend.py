# import os
# import cv2
# import math
# import easyocr
# import numpy as np
# from openai import OpenAI
# from dotenv import load_dotenv
# from PIL import Image, ImageDraw, ImageFont
#
# # Load environment variables
# load_dotenv()
#
#
# class MacStableTranslator:
#     def __init__(self):
#         # 1. Initialize OpenAI
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY not found in .env")
#         self.client = OpenAI(api_key=api_key)
#
#         print("[1/4] Initializing Engine (EasyOCR for Mac)...")
#         # EasyOCR is PyTorch-based and stable on Apple Silicon
#         self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
#
#     def get_colors(self, img, box):
#         """Extracts background color and decides text color (black/white)."""
#         # EasyOCR returns box as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
#         pts = np.array(box, dtype=np.int32)
#
#         # Create mask
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         cv2.fillPoly(mask, [pts], 255)
#
#         # Calculate mean color
#         bg_mean = cv2.mean(img, mask=mask)[:3]
#
#         # Determine brightness for contrast
#         brightness = (bg_mean[0] * 0.299 + bg_mean[1] * 0.587 + bg_mean[2] * 0.114)
#         text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
#
#         return tuple(map(int, bg_mean)), text_color
#
#     def calculate_angle(self, box):
#         """Calculates rotation angle of the text box."""
#         (tl, tr, br, bl) = box
#         # Calculate angle of the top edge
#         delta_y = tr[1] - tl[1]
#         delta_x = tr[0] - tl[0]
#         angle = math.degrees(math.atan2(delta_y, delta_x))
#         return angle
#
#     def translate_batch(self, texts, target_lang):
#         """Sends all text to OpenAI in one go for context."""
#         if not texts: return []
#         print(f"Translating {len(texts)} text blocks...")
#
#         prompt = (f"Translate these technical terms into {target_lang}. "
#                   "Return ONLY the translations separated by newlines. "
#                   "Maintain the exact order. Do not translate brand names.")
#
#         response = self.client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a technical translator."},
#                 {"role": "user", "content": prompt + "\n" + "\n".join(texts)}
#             ],
#             temperature=0.3
#         )
#         return response.choices[0].message.content.strip().split('\n')
#
#     def process_image(self, image_path, target_lang="Spanish"):
#         print(f"--- Processing: {image_path} ---")
#
#         # 1. Load Image
#         img = cv2.imread(image_path)
#         if img is None:
#             print("Error: Image not found.");
#             return
#
#         # 2. Detect Text
#         # detail=1 gives us bounding box coordinates
#         results = self.reader.readtext(img)
#
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         original_texts = []
#         metadata = []
#
#         print(f"Detected {len(results)} text regions.")
#
#         for (box, text, prob) in results:
#             # Filter low confidence or tiny text
#             if prob < 0.40 or len(text.strip()) < 2:
#                 continue
#
#             original_texts.append(text)
#
#             # Prepare Mask for Inpainting
#             pts = np.array(box, dtype=np.int32)
#             cv2.fillPoly(mask, [pts], 255)
#
#             # Get Colors & Angle
#             bg_c, txt_c = self.get_colors(img, box)
#             angle = self.calculate_angle(box)
#
#             metadata.append({
#                 'box': box,
#                 'color': txt_c,
#                 'angle': angle,
#                 'width': np.linalg.norm(np.array(box[1]) - np.array(box[0])),
#                 'height': np.linalg.norm(np.array(box[3]) - np.array(box[0]))
#             })
#
#         # 3. Translate
#         translated_texts = self.translate_batch(original_texts, target_lang)
#
#         # 4. Inpaint (Clean Background)
#         print("Cleaning background...")
#         # Dilate mask slightly to cover edge artifacts
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=1)
#         clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
#
#         # 5. Render Text
#         pil_img = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(pil_img)
#
#         # Font Setup
#         try:
#             # Standard Mac font
#             font_path = "/System/Library/Fonts/Helvetica.ttc"
#             # Verify it loads
#             ImageFont.truetype(font_path, 10)
#         except:
#             font_path = None  # Fallback to default
#
#         for i, meta in enumerate(metadata):
#             if i >= len(translated_texts): break
#
#             text = translated_texts[i].strip()
#             box = meta['box']
#             x, y = box[0][0], box[0][1]  # Top-left coordinates
#             target_w = meta['width']
#             target_h = meta['height']
#
#             # --- AUTO-FIT LOGIC ---
#             # Start with a font size that matches the box height
#             font_size = int(target_h * 0.8)
#             if font_size < 10: font_size = 10
#
#             font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
#
#             # Shrink until it fits width
#             while font_size > 8:
#                 # Get text size
#                 if hasattr(draw, "textbbox"):
#                     bbox = draw.textbbox((0, 0), text, font=font)
#                     text_w = bbox[2] - bbox[0]
#                 else:
#                     text_w = draw.textlength(text, font=font)
#
#                 if text_w <= target_w:
#                     break
#
#                 font_size -= 1
#                 if font_path:
#                     font = ImageFont.truetype(font_path, font_size)
#
#             # --- ROTATION LOGIC ---
#             # If the text is rotated (like in your diagram), we draw it on a separate layer and rotate it
#             if abs(meta['angle']) > 5:
#                 # Create a transparent layer
#                 txt_layer = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
#                 d = ImageDraw.Draw(txt_layer)
#                 d.text((x, y), text, font=font, fill=meta['color'] + (255,))
#
#                 # Rotate around the text starting point
#                 rotated_layer = txt_layer.rotate(meta['angle'], center=(x, y))
#                 pil_img.paste(rotated_layer, (0, 0), rotated_layer)
#             else:
#                 # Standard horizontal text
#                 draw.text((x, y), text, font=font, fill=meta['color'])
#
#         # 6. Save
#         output_filename = f"translated_{os.path.basename(image_path)}"
#         pil_img.save(output_filename)
#         print(f"SUCCESS! Image saved to: {output_filename}")
#
#
# if __name__ == "__main__":
#     p = input("Enter Image Path: ").strip()
#     l = input("Target Language: ").strip()
#     if os.path.exists(p):
#         translator = MacStableTranslator()
#         translator.process_image(p, l)
#     else:
#         print("Path not found.")

import os
import cv2
import math
import easyocr
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()


class TransparentTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def get_colors(self, img_bgr, box):
        """ Sample text color from the BGR layers only """
        pts = np.array(box, dtype=np.int32)
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        bg_mean = cv2.mean(img_bgr, mask=mask)[:3]
        brightness = (bg_mean[0] * 0.114 + bg_mean[1] * 0.587 + bg_mean[2] * 0.299)
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        return tuple(map(int, bg_mean)), text_color

    def translate_batch(self, texts, target_lang):
        if not texts: return []
        print(f"Translating {len(texts)} lines...")
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Translate to {target_lang}. Return only lines:\n" + "\n".join(texts)}]
        )
        return resp.choices[0].message.content.strip().split('\n')

    def process(self, image_path, target_lang="Spanish"):
        print(f"--- Processing: {image_path} ---")

        # 1. LOAD WITH TRANSPARENCY (Crucial Step)
        img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None: print("Error: File not found."); return

        # 2. SEPARATE ALPHA CHANNEL
        has_alpha = False
        if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
            # It is BGRA
            b, g, r, a = cv2.split(img_raw)
            img_bgr = cv2.merge([b, g, r])  # Work on this
            alpha_mask = a  # Save this for later
            has_alpha = True
            print("   [Info] Transparency detected and preserved.")
        else:
            img_bgr = img_raw
            print("   [Info] No transparency found (Standard Image).")

        # 3. DETECT & MASK (On Color Layer)
        results = self.reader.readtext(img_bgr)
        inp_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        texts = []
        items = []

        for (box, text, prob) in results:
            if prob < 0.4: continue
            texts.append(text)

            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(inp_mask, [pts], 255)

            _, txt_color = self.get_colors(img_bgr, box)

            # Geometry for fitting
            w = np.linalg.norm(pts[1] - pts[0])
            h = np.linalg.norm(pts[3] - pts[0])
            items.append({'box': box, 'color': txt_color, 'w': w, 'h': h})

        # 4. TRANSLATE
        translated = self.translate_batch(texts, target_lang)

        # 5. CLEAN TEXT (Inpaint on Color Layer Only)
        kernel = np.ones((3, 3), np.uint8)
        inp_mask = cv2.dilate(inp_mask, kernel, iterations=1)
        clean_bgr = cv2.inpaint(img_bgr, inp_mask, 3, cv2.INPAINT_TELEA)

        # 6. RECOMBINE LAYERS
        if has_alpha:
            # Merge cleaned colors + original alpha
            final_bgra = cv2.merge([clean_bgr[:, :, 0], clean_bgr[:, :, 1], clean_bgr[:, :, 2], alpha_mask])
            # Convert BGRA -> RGBA for PIL
            pil_img = Image.fromarray(cv2.cvtColor(final_bgra, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(pil_img)

        # 7. RENDER TEXT
        try:
            font_path = "/System/Library/Fonts/Helvetica.ttc"
        except:
            font_path = None

        for i, item in enumerate(items):
            if i >= len(translated): break
            text = translated[i].replace('"', '').strip()
            box = item['box']
            x, y = box[0][0], box[0][1]

            # Fit Text
            fontsize = int(item['h'] * 0.9)
            font = ImageFont.truetype(font_path, fontsize) if font_path else ImageFont.load_default()

            while fontsize > 8:
                if hasattr(draw, "textbbox"):
                    w = draw.textbbox((0, 0), text, font=font)[2]
                else:
                    w = draw.textlength(text, font=font)
                if w <= item['w']: break
                fontsize -= 1
                if font_path: font = ImageFont.truetype(font_path, fontsize)

            draw.text((x, y), text, font=font, fill=item['color'])

        # 8. SAVE AS PNG (Required for Transparency)
        out_name = f"transparent_{os.path.basename(image_path)}"
        if not out_name.endswith(".png"):
            out_name = os.path.splitext(out_name)[0] + ".png"

        pil_img.save(out_name, "PNG")
        print(f"   Success! Saved to: {out_name}")


if __name__ == "__main__":
    p = input("Image Path: ").strip()
    l = input("Language: ").strip()
    if os.path.exists(p):
        TransparentTranslator().process(p, l)
    else:
        print("File not found.")