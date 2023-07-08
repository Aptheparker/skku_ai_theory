# -*- coding: utf-8 -*-
# *** Do not modify the code ***

""" Data Load & Prepocessing Classes """

#   *** Do not import any library except already imported libraries ***
import argparse
import json
import nltk
import numpy as np
import math
import random

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pathlib import Path
from sklearn.svm import *
from tqdm import tqdm, trange
from typing import Union, List, Dict, Tuple, Optional

nltk.download('punkt')                      
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
#   *** Do not import any library except already imported libraries ***

class AI_util:
    def __init__(self):
         #   *** Do not modify the code ***
        self.extract_specific_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # Noun, Verb POS tags
        self.stop_words = set(stopwords.words('english'))
        self.word2idx: Dict[str, int]
        self.train_binary: List[Tuple[str, List[float], int]]
        self.test_binary: List[Tuple[str, List[float], int]]
        self.label2idx = {label: idx for idx, label in enumerate(['medical', 'business', 'graphics', 'food', 'entertainment', 'space', 'politics', 'sport', 'historical', 'technologie'])}
         #   *** Do not modify the code ***

    def load_data(self, data_path: Path, data_type: str = 'train') -> List[Tuple[str, List[str], int]]:
         #   *** Do not modify the code ***
        
        tokenized_data = list()
        if data_type == 'train':
            tokens_set = set()
        
        # 1. Load json data file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        for d in tqdm(data, desc='Tokenizing... : '):
            paragraph = d['paragraph']
            tokenized_paragraph = None
        # 2. Preprocess the data paragraph
            pos_tokens = nltk.pos_tag(word_tokenize(paragraph.lower()))
            tokenized_paragraph = [
                token[0] for token in pos_tokens
                if (token[1] in self.extract_specific_tags) and (not token[0] in self.stop_words)
            ]

            if data_type == 'train':
                tokens_set.update(tokenized_paragraph)
            tokenized_data.append((d['id'], tokenized_paragraph, self.label2idx[d['label']]))
        # 3. (if train data) Make word to index dictionary (use *self.word2idx)
        if data_type == 'train':
            self.word2idx = {word: idx for idx, word, in enumerate(sorted(tokens_set))}

        return tokenized_data[:]
         #   *** Do not modify the code ***

    def Calculate_Binary(self):
         #   *** Do not modify the code ***
        return


def save_test_result(result: Dict[str, Union[Tuple[float, float, float, int], Tuple[float, int]]], 
                    std_name: Optional[str] = None, 
                    std_id: Optional[str] = None):
    output_path = Path('./pa2_test_{}_{}.txt'.format(std_name, std_id))
    with open(output_path, mode='w', encoding="utf-8") as f:
        for model_type in ['knn', 'svm', 'mlp']:
            tmp = result[model_type]
            label_name = ['medical', 'business', 'graphics', 'food', 'entertainment', 'space', 'politics', 'sport', 'historical', 'technologie']
            
            headers = ["precision", "recall", "f1-score", "# docs"]
            name_width = max(len(cn) for cn in label_name)
            width = max(name_width, len('micro avg'))
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width) + '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9}\n'
            for label in label_name:
                report += row_fmt.format(
                    label, 
                    tmp[label][0], 
                    tmp[label][1], 
                    tmp[label][2], 
                    tmp[label][3], 
                    width=width
                )
            row_fmt_accuracy = '{:>{width}s} ' + \
                                ' {:>9.2}' * 2 + ' {:>9.2f}' + \
                                ' {:>9}\n'
            report += row_fmt_accuracy.format(
                'accuracy', 
                '', 
                '', 
                tmp['accuracy'][0], 
                tmp['accuracy'][1], 
                width=width
            )

            f.write('Model Type : {}\n'.format(model_type))
            f.write(report + '\n')

# *** Do not modify the code ***
