from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, AutoTokenizer, AddedToken
from google.colab import drive
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import json

import utils

DATASET_PATH = 'dataset.json'

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)


def get_gpt2_tokenizer_with_markers(marker_list):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if len(marker_list) == 0:
        return tokenizer

    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer


def tokenize_function(example):
    inputs = tokenizer(example['perturbed_text'], padding='max_length', truncation=True, max_length=128)
    labels = tokenizer(example['original_text'], padding='max_length', truncation=True, max_length=128)['input_ids']
    inputs['labels'] = labels
    return inputs


if __name__ == '__main__':
    tokenizer = get_gpt2_tokenizer_with_markers(
        [utils.MARKER_HOP_SING, utils.MARKER_HOP_PLUR])
    tokenizer.pad_token = tokenizer.eos_token

    train_data = Dataset.from_list(data['train']).map(tokenize_function, batched=True)
    validate_data = Dataset.from_list(data['validate']).map(tokenize_function, batched=True)

    dataset = DatasetDict({
        'train': train_data,
        'validate': validate_data
    })

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    model.resize_token_embeddings(len(tokenizer))

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validate']
    )

    trainer.train()
