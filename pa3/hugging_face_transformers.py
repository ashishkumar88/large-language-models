# author : Ashish Kumar
# description : This is the main file for the programming assignment 3. This file contains the code for the training and evaluation of the model.


# system imports
import random
from abc import ABC

# library imports
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import matplotlib.pyplot as plt
import evaluate
import numpy as np

# Creat the SAMsum data set loader class
class SAMSumDatasetLoader:
    """Class to load and analyze the SAMSum dataset.
    """

    def __init__(self) -> None:
        self._train_dataset = load_dataset("samsum", split="train[:100%]")
        self._test_dataset = load_dataset("samsum", split="test[:5%]")
        self._val_dataset = load_dataset("samsum", split="validation[:50%]")

    def plot_length_distribution(self) -> None:
        dialogue_lengths = [len(dialogue) for dialogue in self._train_dataset['dialogue']]
        summary_lengths = [len(summary) for summary in self._train_dataset['summary']]

        # create a histogram of the lengths
        plt.figure()
        plt.ion()
        plt.show(block=False)

        plt.title('Dialogue and Summary Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.hist(dialogue_lengths, bins=50, alpha=0.5, label='Dialogue')
        plt.hist(summary_lengths, bins=50, alpha=0.5, label='Summary')
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.001)

    def get_dataset(self) -> dict:
        return self._test_dataset, self._test_dataset, self._val_dataset

# Create an instance of the data set loader and plot the lengths
class SummarizationBase(ABC):
    """Base class for the summarization models.
    """

    def __init__(self, dataset_loader) -> None:
        self._pipeline = None
        self._train_dataset, self._test_dataset, self._validation_dataset = dataset_loader.get_dataset()
        self._rouge = evaluate.load('rouge')

    def summarize_random_dialogues(self, min_length=50, max_length=100, num_tests=3) -> None:

        length_dialogue = len(self._test_dataset['dialogue'])

        # get a random dialogue from the dataset
        random.seed(10)

        print("Model: ", self.__class__.__name__)
        for i in range(num_tests):         
            random_index = random.randint(0, length_dialogue)
            random_dialogue = self._test_dataset['dialogue'][random_index]
            original_summary = self._test_dataset['summary'][random_index]
            output = self._pipeline(random_dialogue, min_length=min_length, max_length=max_length)
            generated_summary = output[0]['summary_text']
            print(f"{str(i + 1)}. \n")
            print("Dialogue: ", random_dialogue)
            print("Generated Summary: ", generated_summary)
            print("Original Summary: ", original_summary)
            print("\n")

    def evaluate(self) -> None:
        """Evaluate the model on the SAMSum dataset.
        """

        length_dialogue = len(self._test_dataset['dialogue'])
        generated_summaries = []

        for i in range(length_dialogue):
            random_dialogue = self._test_dataset['dialogue'][i]
            output = self._pipeline(random_dialogue)
            generated_summary = output[0]['summary_text']
            generated_summaries.append(generated_summary)

        # calculate the rouge scores
        rouge = self._rouge.compute(predictions=generated_summaries, references=self._test_dataset['summary'], use_stemmer=True)
        print("Rouge Scores on test dataset: ", rouge)

class PegasusLargeSummarization(SummarizationBase):
    
    def __init__(self, dataset) -> None:

        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        super().__init__(dataset)
        self._pipeline = pipeline('summarization', model="google/pegasus-large")
    

class BARTLargeSummarization(SummarizationBase):
    
    def __init__(self, dataset) -> None:

        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        super().__init__(dataset)
        
        self._pipeline = pipeline('summarization', model="facebook/bart-large-cnn")


class T5LargeSummarization(SummarizationBase):

    def __init__(self, dataset, model_name=None) -> None:

        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        super().__init__(dataset)
        
        model_name = "t5-large" if model_name is None else model_name   
        
        self._pipeline = pipeline('summarization', model=model_name)

    def summarize_random_dialogues(self, min_length=50, max_length=100, num_tests=3) -> None:

        length_dialogue = len(self._test_dataset['dialogue'])

        # get a random dialogue from the dataset
        random.seed(10)

        print("Model: ", self.__class__.__name__)
        for i in range(num_tests):         
            random_index = random.randint(0, length_dialogue)
            random_dialogue = self._test_dataset['dialogue'][random_index]
            original_summary = self._test_dataset['summary'][random_index]
            random_dialogue = "summarize: " + random_dialogue
            output = self._pipeline(random_dialogue, min_length=min_length, max_length=max_length)
            generated_summary = output[0]['summary_text']
            print(f"{str(i + 1)}. \n")
            print("Dialogue: ", random_dialogue)
            print("Generated Summary: ", generated_summary)
            print("Original Summary: ", original_summary)
            print("\n")

class T5SmallSummarization(T5LargeSummarization):

    def __init__(self, dataset, model_name=None) -> None:

        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        model_name = "t5-small" if model_name is None else model_name   
        
        super().__init__(dataset, model_name)

# The following section defines a class that fine tunes the T5 model
class T5FineTuning:
    """ Class to fine tune the T5 model on the SAMSum dataset.
    """
    
    def __init__(self, 
            dataset_loader,
            max_input_length: int = 512,
            max_target_length: int = 128,
            batch_size: int = 16,
            learning_rate: float = 1e-3,
            num_epoch : int = 8,
            weight_decay: float = 0.01,
            save_total_limit: int = 3,
            model_name: str = "t5-small"
            ) -> None:

        if dataset_loader is None:
            raise ValueError("Dataset cannot be None")
        
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._train_dataset, self._test_dataset, self._validation_dataset = dataset_loader.get_dataset()
        self.fine_tuned_model_name = f"fine-tuned-{self._model_name}"

        # fine tuning parameters
        self._max_input_length = max_input_length
        self._max_target_length = max_target_length

        self._training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{self.fine_tuned_model_name}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            overwrite_output_dir=True,
            logging_dir="./logs",
            learning_rate=learning_rate,
            num_train_epochs=num_epoch,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            gradient_accumulation_steps=16

        )

        self._rouge_metric = evaluate.load("rouge")

    
    def _tokenize(self, samples):
        """ Tokenize the input and output.
        """

        input_dialogues = [f"summarize: {dialogue}" for dialogue in samples['dialogue']]

        # tokenize the input and output
        tokenized_input = self._tokenizer(input_dialogues, max_length=self._max_input_length, truncation=True)
        tokenized_output = self._tokenizer(samples['summary'], max_length=self._max_target_length, truncation=True)

        # replace the input_ids with the labels
        tokenized_input["labels"] = tokenized_output["input_ids"] # labels are used to compute the loss

        return tokenized_input

    def _compute_metrics(self, prediction):
        """ Compute the rouge score.
        """

        predictions, labels = prediction

        # decode the predictions and labels
        predictions = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # replace -100 in the labels as the tokenizer will not decode them
        labels = np.where(labels != -100, labels, self._tokenizer.pad_token_id)

        labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        # compute the rouge score
        rouge_output = self._rouge_metric.compute(predictions=predictions, references=labels, use_stemmer=True)

        prediction_lengths = [np.count_nonzero(prediction != self._tokenizer.pad_token_id) for prediction in predictions]

        rouge_output["prediction_lengths"] = np.mean(prediction_lengths)

        return rouge_output 

    def fine_tune(self):
        """ Fine tune the model on the SAMSum dataset.
        """

        tokenized_train_dataset = self._train_dataset.map(self._tokenize, batched=True)
        tokenized_val_datasest = self._validation_dataset.map(self._tokenize, batched=True)
        data_collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model_name) # dynamic padding       

        self._trainer = Seq2SeqTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=self._training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_datasest,
            data_collator=data_collator,
        )

        self._trainer.train()

    def save_model(self):
        self._trainer.save_model(self.fine_tuned_model_name)


if __name__ == "__main__":

    dataset_loader = SAMSumDatasetLoader() 
    # plot the length distribution of the dataset
    dataset_loader.plot_length_distribution()
        
    # create the models and summarize random dialogues
    num_tests = 2
    pegas_large_model = PegasusLargeSummarization(dataset_loader)
    pegas_large_model.summarize_random_dialogues(num_tests=num_tests)
    pegas_large_model.evaluate()
    del pegas_large_model

    bart_large_model = BARTLargeSummarization(dataset_loader)   
    bart_large_model.summarize_random_dialogues(num_tests=num_tests)
    bart_large_model.evaluate()
    del bart_large_model

    t5_large_model = T5LargeSummarization(dataset_loader)
    t5_large_model.summarize_random_dialogues(num_tests=num_tests)
    t5_large_model.evaluate()
    del t5_large_model

    t5_small_model = T5SmallSummarization(dataset_loader)
    t5_small_model.summarize_random_dialogues(num_tests=num_tests)
    print("Evaluating T5 Small Model before fine-tuning")   
    t5_small_model.evaluate()
    del t5_small_model
        
    if 'pegas_large_model' in globals():
        del pegas_large_model

    if 'bart_large_model' in globals():
        del bart_large_model

    if 't5_large_model' in globals():
        del t5_large_model

    if 't5_small_model' in globals():
        del t5_small_model

    # fine tune the T5 model
    max_input_length = 512
    max_target_length = 128
    batch_size = 16
    learning_rate = 1e-2
    num_epoch = 24
    weight_decay = 0.01
    save_total_limit = 3

    t5_fine_tuning = T5FineTuning(
        dataset_loader,
        max_input_length,
        max_target_length,
        batch_size,
        learning_rate,
        num_epoch,
        weight_decay,
        save_total_limit,
        model_name="t5-small"
    )
    t5_fine_tuning.fine_tune()   
    t5_fine_tuning.save_model() 

    # Test the fine tuned model
    t5_model = T5SmallSummarization(dataset_loader, t5_fine_tuning.fine_tuned_model_name)
    t5_model.summarize_random_dialogues(num_tests=1)
    print("Evaluating T5 Small Model after fine-tuning")
    t5_model.evaluate() # evaluate the model on the test dataset

    # Below is a comparison of summaries from the pre-fine-tuned model and the fine-tuned model
    del t5_model
    num_tests = 10
    print("Evaluating T5 Small Model before fine-tuning")
    t5_model = T5SmallSummarization(dataset_loader)
    t5_model.summarize_random_dialogues(num_tests=num_tests)

    del t5_model
    print("Evaluating T5 Small Model after fine-tuning")
    t5_model = T5SmallSummarization(dataset_loader, t5_fine_tuning.fine_tuned_model_name)
    t5_model.summarize_random_dialogues(num_tests=num_tests)


