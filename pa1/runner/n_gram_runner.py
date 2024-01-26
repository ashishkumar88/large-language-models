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
import argparse
import os
from random import shuffle
import sys

# library imports
from tkinter import Button, END, Entry, Label, Text, Tk, messagebox

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from base_runner import BaseRunner
from data.loader import DataLoader
from data.preprocess import Preprocessor

class NGramRunner(BaseRunner):
    """ This class is responsible for running the ngram training and testing.

    It contains an implementation of the ngram model. In this class, the ngram model is a 
    language model that predicts the next word based on the previous n-1 words. The ngram 
    model is trained on the training dataset and tested on the test dataset. 
    """

    def __init__(self,
        dataset: str = "reuters",
        ngram: int = 1,
        remove_numbers: bool = False,
        train_split: float = 0.8,
        perform_test: bool = False,
        launch_gui: bool = False
    ):
        super().__init__()
        self._dataset = DataLoader.load_dataset(dataset)
        self._ngram = ngram
        self._remove_numbers = remove_numbers
        self._train_split = train_split
        self._initialized = False
        self._perform_test = perform_test
        self._best_word_for_n_minus_one_gram = {}
        self._launch_gui = launch_gui

        if self._ngram < 1:
            raise ValueError("The value error must be greater than 0.")
        else:
            self._initialized = True

    def run(self):
        """ Run the runner. 

        This method is the entry point of the runner. This method is responsible for 
        preprocessing the data, splitting the data into train, validation and test, 
        training the model and testing the model.
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")

        self._dataset, self._vocab = Preprocessor.preprocess_data(self._dataset, self._remove_numbers) # preprocesses the data, type changes from nltk.sentences to list
        shuffle(self._dataset) # shuffles the dataset
        self._split_dataset() # splits the dataset into train and test

        # train the model
        self._train()

        # test the model
        if self._perform_test:
            self._test()

        # launch the GUI
        if self._launch_gui:
            self._launch_user_interface()

    def _split_dataset(self):
        """ Split the dataset into train and test sets
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        dataset_size = len(self._dataset)
        train_size = int(dataset_size * self._train_split)

        self._train_dataset = self._dataset[:train_size]
        self._test_dataset = self._dataset[train_size:]
    
    def _train(self):
        """Train the ngram model
        """

        print("Training the model...")
        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        # create the ngrams and calculate the frequency of each self.
        self._ngrams, self._ngram_freq, self._n_minus_one_grams, self._n_minus_one_gram_freq = self._create_ngrams_with_frequency(self._train_dataset)
        print("Training complete.")

    def _create_ngrams_with_frequency(self, dataset):
        """ Create the ngrams and calculate the frequency of each self.

        Additionally, this method also creates the (n-1)grams and calculates the frequency of each self.
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        ngrams = []
        n_minus_one_grams = []
        for sentence in dataset:
            ngrams += zip(*[sentence[i:] for i in range(self._ngram)])

            if self._ngram > 1:
                n_minus_one_grams += zip(*[sentence[i:] for i in range(self._ngram - 1)])

        # calculate the frequency of each ngram in the dataset
        ngram_freq = self._get_ngram_frequency(ngrams)
        n_minus_one_gram_freq = self._get_ngram_frequency(n_minus_one_grams) if self._ngram > 1 else {}

        return ngrams, ngram_freq, n_minus_one_grams, n_minus_one_gram_freq

    def _get_ngram_frequency(self, ngrams):
        """Get the frequency of the ngrams
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        ngram_freq = {}

        for ngram in ngrams:
            if ngram in ngram_freq:
                ngram_freq[ngram] += 1
            else:
                ngram_freq[ngram] = 1

        return ngram_freq
    
    def _launch_user_interface(self):
        """Launch the GUI
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        print("Launching the GUI...")

        # create the GUI
        self._root = Tk()
        self._root.title("Ngram based sentence generator")
        self._root.geometry("600x400")

        # create the input box
        self._label_input_1 = Label(self._root, text=f"Enter the partial sentence of length {self._ngram - 1}")
        self._label_input_1.grid(row=0, column=0)
        self._input_box_n_minus_gram = Entry(self._root, width=50)
        self._input_box_n_minus_gram.grid(row=0, column=1)

        self._label_input_2 = Label(self._root, text="Enter the length of the sentence")
        self._label_input_2.grid(row=1, column=0)
        self._input_box_sentence_length = Entry(self._root, width=10)
        self._input_box_sentence_length.insert(END, "20")
        self._input_box_sentence_length.grid(row=1, column=1)

        # create the output box
        self._label_output = Label(self._root, text="Generated sentence")
        self._label_output.grid(row=2, column=0)
        self._output_box = Text(self._root, width=50, height=10)
        self._output_box.grid(row=2, column=1)

        # create the generate button
        def generate():
            if self._input_box_n_minus_gram.get().strip() == "":
                messagebox.showerror("Error", "Please enter the partial sentence.")
            else:
                n_minus_one_gram = tuple(self._input_box_n_minus_gram.get().strip().split(" "))
                sentence_length = int(self._input_box_sentence_length.get())

                sentence = self._generate_sentence(n_minus_one_gram, sentence_length)
                self._output_box.delete(1.0, END)
                self._output_box.insert(END, " ".join(sentence))


        self._generate_button = Button(self._root, text="Generate", command=generate)
        self._generate_button.grid(row=3, column=0)

        # create the clear button
        def clear_output():
            self._input_box_n_minus_gram.delete(0, END)
            self._output_box.delete(1.0, END)
            self._input_box_sentence_length.delete(0, END)
            self._input_box_sentence_length.insert(END, "20")

        self._clear_button = Button(self._root, text="Clear", command=clear_output)
        self._clear_button.grid(row=3, column=1)

        self._root.mainloop()

    def _test(self):
        
        print("Testing the model...")
        correct_predictions = 0
        incorrect_predictions = 0

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        total_sentences = len(self._test_dataset)
        sentence_count = 0
        
        for sentence_tokens in self._test_dataset:
            sentence_count += 1
            sentence_length = len(sentence_tokens)

            if sentence_length < self._ngram:
                continue
            else:
                for i in range(sentence_length - self._ngram + 1):
                    sentence = sentence_tokens[i:i+self._ngram-1]
                    next_word = sentence_tokens[i+self._ngram-1]
                    
                    predicted_word = self._generate_next_word(tuple(sentence))

                    if predicted_word == next_word:
                        correct_predictions += 1
                    else:
                        incorrect_predictions += 1

            if sentence_count % 500 == 0:
                print(f"Tested {sentence_count} out of {total_sentences} sentences.", end="\r")

        accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
        print(f"Accuracy: {accuracy}")
    
    def _generate_sentence(self, n_minus_one_gram, sentence_length: int = 20):
        """Generate a sentence of a given length using the ngram model
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        if not isinstance(n_minus_one_gram, tuple):
            raise ValueError("N-1 gram should be a tuple.")
        
        if len(n_minus_one_gram) != self._ngram - 1:
            raise ValueError("The length of the ngram is not equal to the ngram size")
        
        if sentence_length <= len(n_minus_one_gram):
            return " ".join(n_minus_one_gram)
        
        sentence = list(n_minus_one_gram)
        for _ in range(sentence_length - len(n_minus_one_gram)):
            next_word = self._generate_next_word(tuple(sentence[-self._ngram + 1:]))
            sentence.append(next_word)

        if len(sentence) > sentence_length:
            print("The generated sentence is longer than the specified length. Truncating the sentence.") # should not happen
            sentence = sentence[:sentence_length]
        
        return " ".join(sentence)
    
    def _generate_next_word(self, n_minus_one_gram):
        """Generate the next word given the (n − 1)-gram
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        if not isinstance(n_minus_one_gram, tuple):
            raise ValueError("N-1 gram should be a tuple.")
        
        if len(n_minus_one_gram) != self._ngram - 1:
            raise ValueError("The length of the ngram is not equal to the ngram size")
        
        # get the probability of each word following the n-1 gram
        if n_minus_one_gram not in self._best_word_for_n_minus_one_gram:
            max_prob = 0
            max_prob_word = None
            for word in self._vocab:
                prob = self._compute_probability(word, n_minus_one_gram)

                if prob > max_prob:
                    max_prob = prob
                    max_prob_word = word
        
            self._best_word_for_n_minus_one_gram[n_minus_one_gram] = max_prob_word # for faster lookup in the future

        return self._best_word_for_n_minus_one_gram[n_minus_one_gram]

    def _compute_probability(self, word, n_minus_one_gram):
        """Calculates the probability of a word following a given (n − 1)-gram
        """

        if not self._initialized:
            raise ValueError("The class has not been initialized.")
        
        if not isinstance(word, str):
            raise ValueError("Word must be a string.")
        
        if not isinstance(n_minus_one_gram, tuple):
            raise ValueError("N-1 gram should be a tuple.")
        
        n_gram = n_minus_one_gram + (word,)
        if len(n_gram) != self._ngram:
            raise ValueError("The length of the ngram is not equal to the ngram size")

        if self._ngram > 1:
            n_gram_freq = self._ngram_freq[n_gram] if n_gram in self._ngram_freq else 0
            n_minus_one_gram_freq = self._n_minus_one_gram_freq[n_minus_one_gram] if n_minus_one_gram in self._n_minus_one_gram_freq else 0

            prob = (n_gram_freq + 1) / (n_minus_one_gram_freq + len(self._vocab)) # add laplace smoothing
        else:

            # calculate total number of tokens
            total_num_tokes = 0

            for _, value in self._ngram_freq.items():
                total_num_tokes += value

            prob = (self._ngram_freq[n_gram] + 1) / total_num_tokes

        return prob

    def _validate(self):
        raise NotImplementedError
        

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--dataset", help="Select the dataset", default="reuters")
    argparse.add_argument("--ngram", help="Select the ngram", default=2)
    argparse.add_argument("--remove-numbers", help="Remove the numbers", action="store_true", default=False)  
    argparse.add_argument("--train-split", help="Train split", default=0.9)
    argparse.add_argument("--test", help="Test the model", action="store_true", default=False)
    argparse.add_argument("--gui", help="Run the GUI", action="store_true", default=False)

    args = argparse.parse_args()

    runner = NGramRunner(
        dataset=args.dataset, 
        ngram=int(args.ngram),
        remove_numbers=args.remove_numbers,
        train_split=float(args.train_split),
        perform_test=args.test,
        launch_gui=args.gui
        )
    
    runner.run()