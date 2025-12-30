"""
Translation module using OpenAI API for text translation.
Provides literal translation while preserving formatting, punctuation, and casing.
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranslationModule:
    """
    Translation module using OpenAI API.
    Provides literal translation with formatting preservation.
    """
    
    def __init__(self):
        """
        Initialize OpenAI client with API key from environment.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using cost-effective model for translation
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text to target language using OpenAI API.
        Preserves punctuation, casing, and line breaks.
        
        Args:
            text: Text to translate
            target_language: Target language name (e.g., "Spanish", "French")
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        
        prompt = f"""Translate the following text to {target_language}. 
Requirements:
1. Provide a literal, accurate translation
2. Preserve all punctuation marks exactly
3. Preserve the original casing (uppercase/lowercase)
4. Preserve line breaks and spacing
5. Do not paraphrase or add explanations
6. If the text is already in {target_language}, return it unchanged

Text to translate:
{text}

Translation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise translation assistant. Translate text literally while preserving formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, literal translation
                max_tokens=1000
            )
            
            translated = response.choices[0].message.content.strip()
            return translated
        
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on error
    
    def translate_ocr_results(self, ocr_results: List[Dict], target_language: str) -> List[Dict]:
        """
        Translate all text in OCR results.
        
        Args:
            ocr_results: List of OCR result dictionaries with 'text' key
            target_language: Target language name
            
        Returns:
            List of dictionaries with added 'translated_text' and 'expansion_ratio' keys
        """
        translated_results = []
        
        for result in ocr_results:
            original_text = result['text']
            translated_text = self.translate_text(original_text, target_language)
            
            # Calculate expansion ratio
            expansion_ratio = self._calculate_expansion_ratio(original_text, translated_text)
            
            # Create new result with translation
            translated_result = result.copy()
            translated_result['translated_text'] = translated_text
            translated_result['expansion_ratio'] = expansion_ratio
            
            translated_results.append(translated_result)
        
        return translated_results
    
    def _calculate_expansion_ratio(self, original: str, translated: str) -> float:
        """
        Calculate text expansion ratio after translation.
        
        Args:
            original: Original text
            translated: Translated text
            
        Returns:
            Expansion ratio (translated_length / original_length)
        """
        if len(original) == 0:
            return 1.0
        return len(translated) / len(original)

