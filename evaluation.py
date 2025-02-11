import argparse
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import json


def calculate_translation_perplexity(model, tokenizer, dataset):
    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for item in dataset:
            source = item['original_text']
            target = item['perturbed_text']

            inputs = tokenizer(source, return_tensors='pt')
            labels = tokenizer(target, return_tensors='pt').input_ids

            outputs = model(input_ids=inputs.input_ids, labels=labels)

            total_loss += outputs.loss.item() * labels.size(1)
            total_tokens += labels.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()


def calculate_bleu_for_data(data):
    references = [[word_tokenize(item['original_text'])] for item in data]
    predictions = [word_tokenize(item['perturbed_text']) for item in data]

    sentence_scores = [sentence_bleu(ref, pred)
                       for ref, pred in zip(references, predictions)]
    corpus_score = corpus_bleu(references, predictions)

    return {
        'sentence_scores': sentence_scores,
        'corpus_score': corpus_score,
        'average_score': sum(sentence_scores) / len(sentence_scores)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to Test dataset.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Path to Model")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation (PPL/BLEU)")
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.type == 'PPL':
        model = GPT2LMHeadModel.from_pretrained(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)


        perplexities = calculate_translation_perplexity(model, tokenizer, data)
        print('perplexity: ', perplexities)

    elif args.type == 'BLEU':

        print('BLEU: ', calculate_bleu_for_data(data))


