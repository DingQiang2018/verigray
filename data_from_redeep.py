import json

from datasets import load_dataset


with open("/home/dingqiang/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/llama2_7B_response_chunk_verigray.json", 'r') as f:
    redeep_ds = json.load(f)
verigray_ds = load_dataset('json', data_files="/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_20251007.jsonl")['train']

sentences = verigray_ds['sentences']
new_sentences = []
for i in range(len(sentences)):
    response_wise_sentences = []
    for j in range(len(sentences[i])):
        scores = redeep_ds[i]['scores'][j]
        sentence = sentences[i][j]
        sentence['nli_pred'] = 'invalid'
        sentence['nli_note'] = {
            'parameter_knowledge_score': scores['parameter_knowledge_scores'], 
            'prompt_attention_score': scores['prompt_attention_score']
        }
        response_wise_sentences.append(sentence)
    new_sentences.append(response_wise_sentences)
verigray_ds = verigray_ds.remove_columns('sentences')
verigray_ds = verigray_ds.add_column('sentences', new_sentences)
verigray_ds.to_json('redeep_results/verigray_sentence_level_results.jsonl')