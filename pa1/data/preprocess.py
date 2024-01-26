#!/usr/bin/env python

# Copyright 2024 Ashish Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# system imports
import string

# library imports
import nltk 
from nltk import word_tokenize

# local imports

class Preprocessor:

    @staticmethod
    def preprocess_data(data, remove_numbers: bool = False):
        """Preprocess the data
        """

        # remove the punctuations and convert to lower case
        sentences = []
        vocab = set()

        for sentence in data:
            string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—'
            new_sentence = [word.lower() for word in sentence if word not in string.punctuation]

            if remove_numbers:
                new_sentence = [word for word in new_sentence if not word.isnumeric()]

            sentences.append(new_sentence)

            for word in new_sentence:
                vocab.add(word)

        return sentences, vocab