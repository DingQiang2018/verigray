import json
import pickle as pk

from datasets import load_dataset


with open('/home/dingqiang/LLM_Check_Hallucination_Detection/data/scores_fava_annot_llama_2044samp.pkl', 'rb') as f:
    scores, sample_indiv_scores, sample_labels = pk.load(f)

ds = load_dataset('json', data_files='/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_20251007.jsonl')['train']

print('max score', max(sample_indiv_scores['hidden']['Hly20']))
print('min score', min(sample_indiv_scores['hidden']['Hly20']))

sentences = ds['sentences']
sentence_idx = 0
new_sentences = []
for i in range(len(sentences)):
    new_sentences.append([])
    for j in range(len(sentences[i])):
        score = sample_indiv_scores['hidden']['Hly20'][sentence_idx]
        new_sentences[-1].append({
            **sentences[i][j],
            'nli_note': json.dumps([{"uncertainty_score": score}])
        })
        sentence_idx += 1

ds = ds.remove_columns(['sentences'])
ds = ds.add_column('sentences', new_sentences)
ds.to_json('/home/dingqiang/hallucination_detection/llm_check_results/verigray_sentence_level_results.jsonl', lines=True)
