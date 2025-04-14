import os
import sys
import glob
import re
import string
import numpy as np
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from src.predictor.utils import chat_words_dict
from src.predictor.exception import SentimentException
from src.predictor.logger import logging


# Define the TextPreprocessor class
class TextPreprocessor:
    def __init__(self, chat_words_dict):
        self.chat_words_dict = chat_words_dict
        self.stop_words = set(stopwords.words('english'))
        self.spell_checker = SpellChecker()

    # Convert text to lowercase
    def to_lowercase(self, text):
        return text.lower() if isinstance(text, str) else text

    # Remove URLs
    def remove_urls(self, text):
        if isinstance(text, str):
            pattern = re.compile(r'https?://\S+|www\.\S+')
            return pattern.sub('', text)
        return text

    # Remove HTML tags
    def remove_html_tags(self, text):
        if isinstance(text, str):
            pattern = re.compile('<.*?>')
            return re.sub(pattern, '', text)
        return text

    # Replace emojis with text descriptions
    def replace_emojis(self, text):
        if isinstance(text, str):
            text_with_descriptions = emoji.demojize(text)
            # Clean up underscores and extra colons in emoji descriptions
            text_with_descriptions = text_with_descriptions.replace("_", " ").strip(":")
            return text_with_descriptions
        return text

    # Replace chat words with their full form
    def replace_chat_words(self, text):
        if isinstance(text, str):
            chat_words_re = re.compile(r'\b(' + '|'.join(self.chat_words_dict.keys()) + r')\b')
            return chat_words_re.sub(lambda x: self.chat_words_dict[x.group()], text)
        return text
    
    # Correct spelling using PySpellChecker
    def correct_spelling(self, text):
        if isinstance(text, str):
            words = text.split()
            corrected_words = []
            for word in words:
                if word not in self.spell_checker:
                    corrected = self.spell_checker.correction(word)
                    corrected_words.append(corrected if corrected else word)  # Ensure None is not added
                else:
                    corrected_words.append(word)
            return ' '.join(corrected_words)
        return text


    def remove_stopwords(self, text):
        if isinstance(text, str):
            return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])
        return text
    
    def remove_stopwords(self, text):
        if isinstance(text, str):
            words = text.split()
            processed_words = []
            i = 0
            while i < len(words):
                word = words[i].lower()
                if word in {"not", "no", "never"} and i + 1 < len(words):  
                    # Combine negation with the next word
                    negated_word = f"{word}_{words[i+1]}"  
                    processed_words.append(negated_word)
                    i += 2  # Skip the next word (already combined)
                elif word not in self.stop_words:
                    processed_words.append(words[i])  
                    i += 1
                else:
                    i += 1  # Skip stop word

            return ' '.join(processed_words)
    
        return text

    # Remove punctuation
    def remove_punctuation(self, text):
        if isinstance(text, str):
            return text.translate(str.maketrans('', '', string.punctuation))
        return text

    # Pipeline to apply all preprocessing steps
    def preprocess(self, text):
        try:
            logging.info("Preprocessing pipeline launched...")
            text = self.to_lowercase(text)
            text = self.remove_urls(text)
            text = self.remove_html_tags(text)
            text = self.replace_emojis(text)
            text = self.replace_chat_words(text)
            text = self.correct_spelling(text)
            text = self.remove_stopwords(text)
            text = self.remove_punctuation(text)
            logging.info("Preprocessing is completed")
            return text
            

        except Exception as e:
            raise SentimentException(e,sys)
        