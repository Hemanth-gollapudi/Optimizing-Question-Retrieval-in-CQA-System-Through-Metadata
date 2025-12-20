# hmr/lang_pipeline.py
from langdetect import detect
from deep_translator import GoogleTranslator

class LanguagePipeline:
    def __init__(self):
        # deep-translator uses GoogleTranslator with cleaner API
        pass

    def detect_language(self, text):
        """Detect the language of input text"""
        try:
            return detect(text)
        except:
            return "en"  # fallback to English

    def translate_to_english(self, text, lang_code):
        """Translate text from source language to English"""
        if lang_code == "en":
            return text
        try:
            translator = GoogleTranslator(source=lang_code, target='en')
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # fallback to original text

    def translate_from_english(self, text, target_lang):
        """Translate text from English to target language"""
        if target_lang == "en":
            return text
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # fallback to original text
