from copy import deepcopy
import json
import requests

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


ATTRIBUTOR_SERVER_URL = 'http://uar.edward.cheftin.cn/attribute'
model = AutoModelForCausalLM.from_pretrained('/home/dingqiang/QwQ-32B', attn_implementation="flash_attention_2", device_map="auto", load_in_4bit=True) # torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('/home/dingqiang/QwQ-32B')


def get_token_avg_prob(
    prompt: str,
    response: str,
    ranges: list
):
    """get the average log probability of some pieces of response (whose char ranges are specified by ranges)
    """
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    response_encoding = tokenizer(response, return_tensors='pt')
    response_ids  = response_encoding['input_ids']

    input_ids = torch.cat([prompt_ids, response_ids],  dim=-1)
    outputs = model(input_ids=input_ids.cuda())
    logits  =  outputs.logits[0].detach().cpu()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    pred_log_probs = log_probs[torch.arange(len(input_ids[0])-1), input_ids[0, 1:]]
    # translate the ranges to a token mask
    mask = torch.zeros_like(response_ids[0], dtype=torch.bool)
    for i in range(len(response_ids[0])):
        for start, end in ranges:
            if response_encoding.token_to_chars(i).start >= start and response_encoding.token_to_chars(i).end <= end:
                mask[i] = True
                break
    mask = torch.cat([torch.zeros(len(prompt_ids[0])-1, dtype=torch.bool), mask], dim=0)
    avg_prob = pred_log_probs[mask].mean().item()
    return avg_prob


def get_sentence_attribution(
    attribution_data,
    sentence_range: list
):
    """get the ranges of the evidence of a sentence
    """
    input_token_ranges = attribution_data['input_token_ranges']
    output_token_ranges = attribution_data['output_token_ranges']
    token_attributions = attribution_data['token_attributions']
    
    sentence_attribution = set()
    for i in range(len(output_token_ranges)):
        output_start, output_end = output_token_ranges[i]
        if output_start >= sentence_range[0] and output_end <= sentence_range[1]:
            sentence_attribution.update(token_attributions[i])
    sentence_attribution = sorted(sentence_attribution)

    # remove isolated evidence tokens
    new_sentence_attribution = []
    for i in range(len(sentence_attribution)):
        if i == 0 and sentence_attribution[1] == sentence_attribution[0] + 1:
            new_sentence_attribution.append(sentence_attribution[i])
        elif i == len(sentence_attribution) - 1 and sentence_attribution[i] == sentence_attribution[i - 1] + 1:
            new_sentence_attribution.append(sentence_attribution[i])
        elif i > 0 and i < len(sentence_attribution) - 1:
            if sentence_attribution[i] == sentence_attribution[i - 1] + 1 or sentence_attribution[i] == sentence_attribution[i + 1] - 1:
                new_sentence_attribution.append(sentence_attribution[i])
    if new_sentence_attribution != []:
        sentence_attribution = new_sentence_attribution

    # token to ranges
    evidence_ranges = []
    start = 0  # the start of the most recent contiguous span of evidence tokens
    for i in range(len(sentence_attribution)-1):
        if sentence_attribution[i] + 1 != sentence_attribution[i + 1]:
            # if the current token is the last token of a contiguous span
            evidence_ranges.append((input_token_ranges[start][0], input_token_ranges[sentence_attribution[i]][1]))
            start = sentence_attribution[i + 1]
    evidence_ranges.append((input_token_ranges[start][0], input_token_ranges[sentence_attribution[-1]][1]))

    return evidence_ranges


def get_prob(examples):
    """write the probs into sentences
    """
    results = {
        'question': [],
        'answer': [],
        'sentences': [],
    }
    for i in range(len(examples['question'])):
        results['question'].append(examples['question'][i])
        results['answer'].append(examples['answer'][i])
        results['sentences'].append([])
        for j in range(len(examples['sentences'][i])):
            prompt = tokenizer.apply_chat_template(
                f"Summarize this paragraph: {examples['question'][i]}", 
                tokenize=False, 
                add_generation_prompt=True
            ) + ''.join([x['text'] for x in examples['sentences'][i][:j]])
            sentence_prob = get_token_avg_prob(
                prompt=prompt,
                response=examples['sentences'][i][j]['text'],
                ranges=[[0, len(examples['sentences'][i][j]['text'])]]
            )

            # contrastive prob
            contrastive_prompt = tokenizer.apply_chat_template(
                f"Summarize this paragraph: {examples['answer'][i]}",
                tokenize=False,
                add_generation_prompt=True
            )

            ## call attribution server to get the evidence
            headers = {'Content-Type': 'application/json'}
            post_data = {'question': examples['question'][i], 'answer': examples['answer'][i]}
            response = requests.post(ATTRIBUTOR_SERVER_URL, headers=headers, json=post_data)
            attribution_data = json.loads(response.text)

            contrastive_prob = get_token_avg_prob(
                prompt=contrastive_prompt,
                response=examples['question'][i],
                ranges=get_sentence_attribution(attribution_data, examples['sentences'][i][j]['range'])
            )

            sentence = deepcopy(examples['sentences'][i][j])
            sentence['log_prob'] = sentence_prob - contrastive_prob
            results['sentences'][-1].append(sentence)

    return results


ds = load_dataset("json", data_files="/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_2.jsonl")['train']
ds = ds.map(get_prob, batched=True, batch_size=5)

# the correlation between the prob and the label
sentences = ds['sentences']
y = {
    'explicitly supported': [],
    'generally supported': [],
    'not sure': [],
    'fabricated': [],
    'inconsistent': [],
}
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        label = sentences[i][j]['annotation']
        prob = sentences[i][j]['log_prob']
        y[label].append(prob)

plt.figure(figsize=(10, 6))
plt.boxplot([y['explicitly supported'], y['generally supported'], y['not sure'], y['fabricated'], y['inconsistent']],
            labels=['explicitly supported', 'generally supported', 'not sure', 'fabricated', 'inconsistent'])
plt.title('Log Probability of Different Labels')
plt.ylabel('Log Probability')
plt.xlabel('Labels')
plt.grid(True)
plt.savefig('log_prob_boxplot.pdf')
plt.show()