from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset


splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    # chunk_overlap=200,
    length_function=len,
)


def do_split(text):
    """split the text using RecursiveCharacterTextSplitter and return the character ranges
    """
    spans = splitter.split_text(text)
    ranges = []
    for span in spans:
        start_pos = text.index(span)
        end_pos = start_pos + len(span)
        ranges.append((start_pos, end_pos))
    return ranges


def build_source(examples, indices):
    results = {
        'source_id': [],
        'prompt': [],
        'prompt_spans': [],
    }
    for i in range(len(examples['question'])):
        results['prompt'].append(examples['question'][i])
        results['source_id'].append(indices[i])
        results['prompt_spans'].append(do_split(examples['question'][i]))
    return results


def build_response(examples, indices):
    results = {
        'source_id': [],
        'response': [],
        'response_spans': [],
    }
    for i in range(len(examples['answer'])):
        results['response_spans'].append([[s['range'][0], s['range'][1]] for s in examples['sentences'][i]])
        results['response'].append(examples['answer'][i])
        results['source_id'].append(indices[i])
    return results


ds = load_dataset("json", data_files="FaithBench_sentence_level_reannotated_20250916.jsonl")['train']
# build the sources
source_ds = ds.map(
    build_source, 
    with_indices=True,
    batched=True,
    batch_size=100,
    remove_columns=ds.column_names,
)
source_ds.to_json('redeep_data/verigray_source.jsonl')
# build the responses
response_ds = ds.map(
    build_response,
    with_indices=True,
    batched=True,
    batch_size=100,
    remove_columns=ds.column_names,
)
response_ds.to_json('redeep_data/verigray_response.jsonl')