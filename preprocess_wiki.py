import datasets
import pickle
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from tqdm import tqdm
from pathlib import Path



### STEP1: Parse Wikipedia
wiki_dataset = load_dataset(
    './wikipedia_xml',
    language="en", date="20221101",
    cache_dir='/mnt/users/j587yang/wikipedia/datasets',
    split='train'
)

### STEP2: de-duplication
print('Running de-duplication')

atomic_dataset = load_dataset('TREC-AToMiC/AToMiC-Texts-v0.1')
combined = concatenate_datasets([atomic_dataset[split] for split in atomic_dataset.keys()])

def get_key(example):
    return {'key': '\t'.join([example['page_title'].lower(), example['section_title'].lower()])}

if not Path('duplicates.pkl').exists():
    duplicate_dict = {}
    positive_keys = set()
    reduced_dict = {}
    for example in tqdm(combined, desc='build duplicate dict ...'):
        key = get_key(example)['key']

        if key not in positive_keys:
            reduced_dict[key] = example['text_id']
            positive_keys.add(key)
        else:
            duplicate_dict[example['text_id']] = reduced_dict[key]

    with open('duplicates.pkl', 'wb') as f:
        pickle.dump(duplicate_dict, f)
else:
    with open('duplicates.pkl', 'rb') as f:
        duplicate_dict = pickle.load(f)

def dedup(example):
    return bool(example['text_id'] not in duplicate_dict.keys())

combined = combined.filter(dedup, num_proc=32)

### STEP3: Get page intersection b/w WIT and Parsed Wikipedia
print('Preparing data projection')
combined = combined.map(get_key, num_proc=32)
wit_keys = set(combined.unique('key'))

if not Path('key2id.pkl').exists():
    key_to_id = {}
    for item in tqdm(combined, desc='building key2id dict'):
        key_to_id[item['key']] = item['text_id']
    with open('key2id.pkl', 'wb') as f:
        pickle.dump(key_to_id, f)
else:
    with open('key2id.pkl', 'rb') as f:
        key_to_id = pickle.load(f)

atomic_pages = set(combined.unique('page_title'))

def valid_page(example):
    return bool(example['title'] in atomic_pages)

print(' ==== Get intersection of WIT and projected Wikipedia')
selected = wiki_dataset.filter(valid_page, num_proc=32)
print(f'Num of pages in the intersection: {len(selected)}')
print(selected[199527])


def data_generator(examples):
    for example in examples:
        _id = example['id']
        page_title = example['title']
        page_url = example['url']
        infobox_media = example['infobox_media']
        for s_id, section_title in enumerate(example['section_title']):
            text = example['text'][s_id]  # string
            media = example['media'][s_id] # list
            category = example['category'] # list

            if s_id == 0:
                context_page = ""
                media = infobox_media
            else:
                context_page = example['text'][0]
            
            if section_title == "External links":
                continue
                
            if len(text) < 1:
                continue
            
            key = '\t'.join([page_title.lower(), section_title.lower()])
            
            new_id = f'projected-{int(_id):08d}-{int(s_id):03d}'
            
            if key in wit_keys:
                old_id = key_to_id[key]
            else:
                old_id = ''
                
            yield {
                "text_id": new_id,
                "old_id": old_id,
                "page_url": page_url,
                "page_title": page_title,
                "section_title": section_title,
                "context_page_description": context_page,
                "context_section_description": text,
                "media": media, 
                "category": category,
            }
            

flatten = Dataset.from_generator(
    data_generator,
    gen_kwargs={'examples': selected},
    features=datasets.Features(
                {
                    "text_id": datasets.Value("string"),
                    "old_id": datasets.Value("string"),
                    "page_url": datasets.Value("string"),
                    "page_title": datasets.Value("string"),
                    "section_title": datasets.Value("string"),
                    "context_page_description": datasets.Value("string"),
                    "context_section_description": datasets.Value("string"),
                    "media": datasets.Sequence(feature=datasets.Value("string")),
                    "category": datasets.Sequence(feature=datasets.Value("string")),
                }
    ),
)
print(flatten)
print(flatten[9527])
flatten.push_to_hub('justram/AToMiC-Map-Texts', split='train')
