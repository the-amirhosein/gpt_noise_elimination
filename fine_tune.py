from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, AutoTokenizer, AddedToken
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
import json
import matplotlib.pyplot as plt

import utils

DATASET_PATH = 'dataset.json'

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)


def get_gpt2_tokenizer_with_markers():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [
        AddedToken(utils.MARKER_HOP_SING, lstrip=True, rstrip=False),
        AddedToken(utils.MARKER_HOP_PLUR, lstrip=True, rstrip=False)
    ]
    tokenizer.add_tokens(new_tokens)
    return tokenizer


def tokenize_function(examples):
   inputs = tokenizer(examples['perturbed_text'], padding='max_length', truncation=True, max_length=128)
   labels = tokenizer(examples['original_text'], padding='max_length', truncation=True, max_length=128)['input_ids']
   inputs['labels'] = labels
   return inputs


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_perplexities = []
        self.eval_perplexities = []

    def compute_perplexity(self, dataset):
        total_loss = 0
        total_length = 0

        for batch in dataset:
            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'].to(self.model.device),
                                     labels=batch['labels'].to(self.model.device))
                total_loss += outputs.loss.item() * batch['input_ids'].size(1)
                total_length += batch['input_ids'].size(1)

        return torch.exp(torch.tensor(total_loss / total_length)).item()

    def on_epoch_end(self, args, state, control, **kwargs):
        train_perplexity = self.compute_perplexity(self.train_dataset)
        eval_perplexity = self.compute_perplexity(self.eval_dataset)

        self.train_perplexities.append(train_perplexity)
        self.eval_perplexities.append(eval_perplexity)

        print(f"Epoch {state.epoch}:")
        print(f"Train Perplexity: {train_perplexity:.2f}")
        print(f"Eval Perplexity: {eval_perplexity:.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_perplexities, label='Train', marker='o')
        plt.plot(self.eval_perplexities, label='Validation', marker='s')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'perplexity_plot_epoch_{state.epoch}.png')
        plt.close()

        return super().on_epoch_end(args, state, control, **kwargs)


if __name__ == '__main__':

    tokenizer = get_gpt2_tokenizer_with_markers()
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))


    train_data = Dataset.from_list(data['train']).map(tokenize_function, batched=True)
    validate_data = Dataset.from_list(data['validate']).map(tokenize_function, batched=True)

    dataset = DatasetDict({
        'train': train_data,
        'validate': validate_data
    })

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validate']
    )

    trainer.train()
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')