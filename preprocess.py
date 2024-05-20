import sys
import json
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
import spacy
spacy.cli.download('en_core_web_sm')
import contextualSpellCheck
import string


class LoadData:
    def __init__(self, path):
        self.path = path
        self.src = self._read_file(f'{self.path}.src')
        self.tgt = self._read_file(f'{self.path}.tgt')
        self.data_len = len(self.src)
        self.data = pd.DataFrame()

    @staticmethod
    def _read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()

    def get_raw_data(self):
        for i in range(self.data_len):
            n = len(self.data)
            self.data.loc[n, 'summary'] = self.src[i]
            self.data.loc[n, 'story'] = self.tgt[i]
        if self.path == 'data/train':
            self.data = self.data[:8000]
        self.data.to_csv(f'{self.path}_en_raw.csv')
        return self.data


class Preprocess:
    def __init__(self, path):
        self.path = path
        self.raw_data = LoadData(self.path).get_raw_data()
        self.data_len = len(self.raw_data)
        self.nlp = spacy.load('en_core_web_sm')
        contextualSpellCheck.add_to_pipe(self.nlp)
        self.data = pd.DataFrame()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def _remove_parentheses_at_start(text):
        text = re.sub(r'^\([^()]*\)', '', text)
        text = text.strip()
        return text

    @staticmethod
    def _standardise_punctuation(text):
        text = text.replace("``", '"')
        text = text.replace("''", '"')
        text = text.replace("*:", '"')
        return text
        
    @staticmethod
    def _capitalise_sentence_start(text):
        text = re.sub(r'(?<!\.\s)([a-zA-Z])', lambda x: x.group().lower(), text)
        text = re.sub(r'(?:^|\. *)(\w)', lambda x: x.group().upper(), text)
        text1 = text[text.find('"') + 1:text.rfind('"')].strip()
        text2 = re.sub(r'(?<!\.\s)([a-zA-Z])', lambda x: x.group().lower(), text1)
        text2 = re.sub(r'(?:^|\. *)(\w)', lambda x: x.group().upper(), text2)
        text = text.replace(text1, text2)
        return text
        
    def _capitalise_proper_nouns(self, text):
        doc = self.nlp(text)
        new_tokens = [token.text.title() if token.pos_=='PROPN' else token.text for token in doc]
        text = ' '.join(new_tokens)
        return text
        
    @staticmethod
    def _correct_punctuation_spacing(text):
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'"\s+', '" ', text)
        text = re.sub(r'\s+"', ' "', text)
        text = re.sub(r'([.,!?;:]")([^\s])', r'\1 \2', text)
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
        return text


    @staticmethod
    def _correct_whitespace(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def _end_of_sentence_punctuation(text):
        if text[-1] in string.punctuation:
            return text
        else:
            return text+'.'

    def _spell_correct(self, text):
        doc = self.nlp(text)
        if doc._.performed_spellCheck:
            text = doc._.outcome_spellCheck
        return text

    def _preprocessing_pipeline(self, text):
        text = self._remove_parentheses_at_start(text)
        text = self._standardise_punctuation(text)
        text = self._capitalise_sentence_start(text)
        text = self._capitalise_proper_nouns(text)
        text = self._correct_punctuation_spacing(text)
        text = self._correct_whitespace(text)
        text = self._end_of_sentence_punctuation(text)
        text = self._spell_correct(text)
        text = ' \n '.join(self.tokenizer.tokenize(text))
        return text

    def preprocess(self):
        for i in range(self.data_len):
            n = len(self.data)
            self.data.loc[n, 'summary'] = self._preprocessing_pipeline(self.raw_data.loc[i, 'summary'])
            self.data.loc[n, 'story'] = self._preprocessing_pipeline(self.raw_data.loc[i, 'story'])
        self.data.to_csv(f'{self.path}_en_processed.csv')
        return self.data