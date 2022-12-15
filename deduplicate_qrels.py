import datasets
import pickle
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from tqdm import tqdm


split = 'test'

qrels = load_dataset(
    'csv',
    data_files=f'datasets/AToMiC-v0.1.0/qrels/{split}.qrels.t2i.trec',
    delimiter=' ',
    column_names=['text_id', 'Q0', 'image_id', 'rel'],
    split='train',
)

with open('src/duplicates.pkl', 'rb') as f:
    duplicate_dict = pickle.load(f)

def dedup(example):
    if example['text_id'] in duplicate_dict.keys():
        return {'text_id': duplicate_dict[example['text_id']]}
    else:
        return {'text_id': example['text_id']}

deduped_qrels = qrels.map(dedup, num_proc=32)

print(len(qrels.unique('text_id')))
print(len(deduped_qrels.unique('text_id')))

deduped_qrels.push_to_hub('justram/AToMiC-Qrels-Dedupe', split=split)