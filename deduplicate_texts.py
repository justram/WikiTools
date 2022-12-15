import datasets
import pickle
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from tqdm import tqdm


atomic_dataset = load_dataset('TREC-AToMiC/AToMiC-Texts-v0.1')
combined = concatenate_datasets([atomic_dataset[split] for split in atomic_dataset.keys()])

def get_key(example):
    return '\t'.join([example['page_title'].lower(), example['section_title'].lower()])

duplicate_dict = {}
positive_keys = set()
reduced_dict = {}
for example in tqdm(combined, desc='build duplicate dict ...'):
    key = get_key(example)

    if key not in positive_keys:
        reduced_dict[key] = example['text_id']
        positive_keys.add(key)
    else:
        duplicate_dict[example['text_id']] = reduced_dict[key]

with open('duplicates.pkl', 'wb') as f:
    pickle.dump(duplicate_dict, f)

def dedup(example):
    return bool(example['text_id'] not in duplicate_dict.keys())

for split in atomic_dataset.keys():
    new = atomic_dataset[split].filter(dedup, num_proc=32)
    new.push_to_hub('justram/AToMiC-Texts-Dedup', split=split)