from datasets import load_dataset


def data_to_fava_annotated(examples):
    results = {
        'prompt': [],
        'output': [],
        'annotated': [],
        'model': []
    }
    for i in range(len(examples['question'])):
        prompt = f"You are a chat bot answering questions using data. You must stick to the answers provided solely by the text in the passage provided. You are asked the question 'Provide a concise summary of the following passage, covering the core pieces of information described.' {examples['question'][i]}"
        answer = examples['answer'][i]
        model = examples['generator'][i]
        for j in range(len(examples['sentences'][i])):
            sentence_range = examples['sentences'][i][j]['range']
            output = answer[sentence_range[0]: sentence_range[1]]
            annotated = f"<subjective>{output}</subjective>" if examples['sentences'][i][j]['annotation'] in ['fabricated', 'contradicting'] else output
            results['prompt'].append(prompt)
            results['output'].append(output)
            results['annotated'].append(annotated)
            results['model'].append(model)
    return results


ds = load_dataset('json', data_files='/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_20251007.jsonl')['train']
fava_annotated_ds = ds.map(data_to_fava_annotated, batched=True, remove_columns=ds.column_names)
fava_annotated_ds.to_json('/home/dingqiang/LLM_Check_Hallucination_Detection/verigray_fava_annotated.json')