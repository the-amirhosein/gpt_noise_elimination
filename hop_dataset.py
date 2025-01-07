from utils import perturb_hop_words4, perturb_hop_tokens4, perturb_hop_control
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def apply_hop(text, hop_type):
    if hop_type == 'control':
        return perturb_hop_control(text)
    elif hop_type == 'token4':
        return perturb_hop_tokens4(text)
    elif hop_type == 'word4':
        return perturb_hop_words4(text)
    else:
        raise ValueError("Invalid hop type. Choose from 'control', 'token4', or 'word4'.")


f = open('/Users/amirhosein/PycharmProjects/mission-impossible-language-models/train_10M/switchboard_parsed.json')
data = json.load(f)
f.close()

results = []
for line in tqdm(data):
    for sent in line['sent_annotations']:
        results.append(sent)

output_data = []
for sent in results:
    perturbed_text, original_text = apply_hop(sent, 'word4')
    res = {
        'original_text': original_text,
        'perturbed_text': perturbed_text
    }
    output_data.append(res)

train_data, validate_data = train_test_split(output_data, test_size=0.2, random_state=42)

dataset = {
    "train": train_data,
    "validate": validate_data
}

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

