from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline, T5Tokenizer, T5ForConditionalGeneration, MarianTokenizer, MarianMTModel, AutoModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate import meteor_score
import time
import torch
import numpy as np
from deep_translator import GoogleTranslator


class Translate:
    def __init__(self, path, data, model_name):
        self.data = data
        self.path = path
        self.len_data = len(self.data)
        self.translated_data = pd.DataFrame()
        self.label_translated_data = pd.DataFrame()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.translator = pipeline("translation", model=self.model, tokenizer=self.tokenizer)
        self.label_translator = GoogleTranslator(source='en', target='tr')

    def _translate_sentence(self, sentence):
        sentences = sentence.split('\n')
        result=''
        for i in range(len(sentences)):
            if(len(sentences[i]) != 0):
                result += self.translator(sentences[i])[0]['translation_text']
            if(i != len(sentences)-1):
                result+='\n'
        return result

    def translate(self):
        translated_data = {}
        for i in range(self.len_data):
            n = len(self.translated_data)
            if self.data.loc[i, 'summary'] in translated_data.keys():
                translated_sum = translated_data[self.data.loc[i, 'summary']]
            else:
                translated_sum = self._translate_sentence(self.data.loc[i, 'summary'])
                translated_data[self.data.loc[i, 'summary']] = translated_sum
            if self.data.loc[i, 'story'] in translated_data.keys():
                translated_st = translated_data[self.data.loc[i, 'story']]
            else:
                translated_st = self._translate_sentence(self.data.loc[i, 'story'])
                translated_data[self.data.loc[i, 'story']] = translated_st
            self.translated_data.loc[n, 'ozet'] = translated_sum
            self.translated_data.loc[n, 'hikaye'] = translated_st
        self.translated_data.to_csv(f'{self.path}_tr.csv')
        return self.translated_data

    def translate_label(self):
        translated_data = {}
        for i in range(self.len_data):
            n = len(self.label_translated_data)
            if self.data.loc[i, 'summary'] in translated_data.keys():
                translated_sum = translated_data[self.data.loc[i, 'summary']]
            else:
                translated_sum = self.label_translator.translate(self.data.loc[i, 'summary'])
                translated_data[self.data.loc[i, 'summary']] = translated_sum
            if self.data.loc[i, 'story'] in translated_data.keys():
                translated_st = translated_data[self.data.loc[i, 'story']]
            else:
                translated_st = self.label_translator.translat(self.data.loc[i, 'story'])
                translated_data[self.data.loc[i, 'story']] = translated_st
            self.label_translated_data.loc[n, 'ozet'] = translated_sum
            self.label_translated_data.loc[n, 'hikaye'] = translated_st
        self.label_translated_data.to_csv(f'{self.path}_tr_label.csv')
        return self.label_translated_data

    def _evaluate_meteor(self, column):
        output_text = [self.translated_data.loc[i, column] for i in range(self.len_data)]
        reference_text = [self.label_translated_data.loc[i, column] for i in range(self.len_data)]
        return meteor_score.meteor_score(reference_text, output_text)

    def _evaluate_bleu(self, column):
        output_text = [self.translated_data.loc[i, column] for i in range(self.len_data)]
        reference_text = [self.label_translated_data.loc[i, column] for i in range(self.len_data)]
        bleu_dict = {}
        bleu_dict['corpus_non-cumulative'] = corpus_bleu(reference_text, output_text)
        bleu_dict['corpus_1-grams'] = corpus_bleu(reference_text, output_text, weights=(1.0, 0, 0, 0))
        bleu_dict['corpus_1-2-grams'] = corpus_bleu(reference_text, output_text, weights=(0.5, 0.5, 0, 0))
        bleu_dict['corpus_1-3-grams'] = corpus_bleu(reference_text, output_text, weights=(0.3, 0.3, 0.3, 0))
        bleu_dict['corpus_1-4-grams'] = corpus_bleu(reference_text, output_text, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_dict['sentence_non-cumulative'] = sentence_bleu(reference_text, output_text)
        bleu_dict['sentence_1-grams'] = sentence_bleu(reference_text, output_text, weights=(1.0, 0, 0, 0))
        bleu_dict['sentence_1-2-grams'] = sentence_bleu(reference_text, output_text, weights=(0.5, 0.5, 0, 0))
        bleu_dict['sentence_1-3-grams'] = sentence_bleu(reference_text, output_text, weights=(0.3, 0.3, 0.3, 0))
        bleu_dict['sentence_1-4-grams'] = sentence_bleu(reference_text, output_text, weights=(0.25, 0.25, 0.25, 0.25))
        return bleu_dict

    def evaluate(self):
        evaluation_dict = {}
        evaluation_dict['summary'] = self._evaluate_bleu('ozet')
        evaluation_dict['summary']['meteor'] = self._evaluate_meteor('ozet')
        evaluation_dict['story'] = self._evaluate_bleu('hikaye')
        evaluation_dict['story']['meteor'] = self._evaluate_meteor('hikaye')
        return evaluation_dict