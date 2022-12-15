import datasets
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset

wiki_dataset = load_dataset(
    './wikipedia_xml',
    language="en", date="20221101",
    cache_dir='/mnt/users/j587yang/wikipedia/datasets',
    split='train'
)

atomic_dataset = load_dataset('TREC-AToMiC/AToMiC-Texts-v0.1')
combined = concatenate_datasets([atomic_dataset[split] for split in atomic_dataset.keys()])
atomic_pages = set(combined.unique('page_title'))

def get_key(example):
    return {'key': '\t'.join([example['page_title'].lower(), example['section_title'].lower()])}

combined = combined.map(get_key, num_proc=32)
positive_keys = set(combined.unique('key'))

# select the pages
def valid_page(example):
    return bool(example['title'] in atomic_pages)

print('filtering')
selected = wiki_dataset.filter(valid_page, num_proc=32)
print(f'Num of pages (filtered): {len(selected)}')
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
            category = example['category'][s_id] # list

            if s_id == 0:
                context_page = ""
                media = infobox_media
            else:
                context_page = example['text'][0]
            
            # skip positive instances
            if '\t'.join([page_title.lower(), section_title.lower()]) in positive_keys:
                continue
            if section_title == "External links":
                continue
            if len(text) < 1:
                continue

            # only select sections w/o media
            if len(media) == 0:
                yield {
                    "text_id": f'neg-{_id}-{s_id}',
                    "page_url": page_url,
                    "page_title": page_title,
                    "section_title": section_title,
                    "hierarchical_section_title": "",
                    "context_page_description": context_page,
                    "context_section_description": text,
                    "category": category,
                }

# generator = data_generator(selected)
# print(next(generator))

flatten = Dataset.from_generator(
    data_generator,
    gen_kwargs={'examples': selected},
    features=datasets.Features(
                {
                    "text_id": datasets.Value("string"),
                    "page_url": datasets.Value("string"),
                    "page_title": datasets.Value("string"),
                    "section_title": datasets.Value("string"),
                    "hierarchical_section_title": datasets.Value("string"),
                    "context_page_description": datasets.Value("string"),
                    "context_section_description": datasets.Value("string"),
                    "category": datasets.Sequence(feature=datasets.Value("string")),
                }
    ),
)
print(flatten)
print(flatten[9527])
flatten.push_to_hub('justram/AToMiC-Neg-Texts', split='train')
