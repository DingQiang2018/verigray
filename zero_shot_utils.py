import os
from time import sleep
import math
import json
import re

from openai import OpenAI
from openai.types.completion_choice import CompletionChoice
from transformers import AutoTokenizer


zero_shot_type = 'vanilla'  # 'sequential_early_exit', 'all_sentences'
pred2idx = {
    'explicitly-supported': 6,
    'generally-supported': 5,
    'out-dependent': 4,
    'ambiguous': 3,
    'fabricated': 2,
    'inconsistent': 1,
    'no-fact': 0,
    'a': 6,
    'b': 5,
    'c': 1,
    'd': 2,
    'e': 4,
    'f': 3,
    'g': 0,
    'a. explicitly-supported': 6,
    'b. generally-supported': 5,
    'c. inconsistent': 1,
    'd. fabricated': 2,
    'e. out-dependent': 4,
    'f. ambiguous': 3,
    'g. no-fact': 0,
}
idx2pred = {
    6: 'explicitly-supported',
    5: 'generally-supported',
    4: 'out-dependent',
    3: 'ambiguous',
    2: 'fabricated',
    1: 'inconsistent',
    0: 'no-fact',
}
openrouter2huggingface = {
    'deepseek/deepseek-r1': '/home/dingqiang/DeepSeek-R1',
    'deepseek/deepseek-r1:free': '/home/dingqiang/DeepSeek-R1',
    'deepseek/deepseek-r1-0528': '/home/dingqiang/DeepSeek-R1',
    'deepseek/deepseek-r1-0528:free': '/home/dingqiang/DeepSeek-R1',
    'deepseek/deepseek-r1-distill-llama-70b:free': '/home/dingqiang/DeepSeek-R1-Distill-Llama-70B',
}


def _build_zero_shot_prompt(document, summary, sentence):
    return f"""You are judging the faithfulness of the assigned sentence of a summary to the source document. The faithfulness has five options: 
A. Explicitly-Supported: all atomic facts of the sentence appear verbatim (up to a lexical or syntactic transformation) within the document.
B. Generally-Supported: the document entails the sentence, and the sentence is not explicitly-supported by the document. Note that some difference between part of the sentence and the document is allowed, but it should be restricted to the case in which the part of the sentence adopts a weaker or less certain utterance than the document. If any part of the sentence adopts a stronger or more certain utterance than the document, consider option D (Fabricated).
C. Inconsistent: the sentence logically contradicts the document.
D. Fabricated: the sentence does not logically contradict the document, and is neither logically implied by the document nor by external world knowledge.
E. Out-Dependent: the sentence is not logically implied by the document but by the union of the document and external world knowledge.
F. Ambiguous: the sentence or the document has multiple interpretations.
G. No-Fact: the sentence is devoid of facts.

Your task is to decompose the target sentence into atomic facts and then output the correct option.

[Document] {document}

[Summary] {summary}

[Sentence in the summary] {sentence}

Your option:
"""


def _build_zero_shot_prompt_all_sentences(document, summary, sentence_ranges):
    sentence_split = ""
    for i, (a, b) in enumerate(sentence_ranges):
        sentence_split += f"[Sentence {i + 1} in the summary] {summary[a:b]}"
        if i < len(sentence_ranges) - 1:
            sentence_split += "\n\n"
    """prompting the model to output all sentences' annotations in one response
    """
    return f"""You are judging the faithfulness of each sentence of a summary to the source document. The faithfulness has five options: 
A. Explicitly supported: Each word in the sentence has verbatim or synonymous evidence in the document.
B. Generally supported: The sentence is indicated by the document but some word of the sentence does not have verbatim or synonymous evidence in the document. Note that some difference between part of the sentence and the document is allowed, but it should be restricted to the case in which the part of the sentence adopts a weaker or less certain utterance than the document. If any part of the sentence adopts a stronger or more certain utterance than the document, consider option D (Fabricated).
C. Inconsistent: The sentence contradicts the document.
D. Fabricated: The sentence is neither entailed nor contradicted by the document.
E. Not sure: Choose this option whenever you are not confident to make a decision, e.g., when people with different knowledge backgrounds may have different decisions on this sentence

[Document] {document}

[Summary] {summary}

The sentence split of the summary is listed as follows:
{sentence_split}

The output should be a list of dictionaries in JSON format, where each dictionary should have the following keys:
- text: the text of the sentece
- faithfulness: the faithfulness of the sentence
"""


def complete_with_retry(
    client: OpenAI,
    nli_model,
    prompt,
    max_tokens,
    timeout=None,
    stop=None,
    logprobs=False,
):
    """retry whenever the request fails due to timeout
    the request will be sent by client.completions.create
    
    Returns
    -------
    str, the response
    str, the stop reason if available
    """
    retry_count = 0
    while True:
        try:
            response = client.completions.create(
                model=nli_model,
                prompt=prompt,
                temperature=0.6,
                max_tokens=max_tokens,
                logprobs=logprobs,
                timeout=timeout,
                stop=stop
            ).choices[0]
            stop_reason = response.stop_reason if response.finish_reason == 'stop' and hasattr(response, 'stop_reason') else None
            if hasattr(response, 'reasoning') and response.reasoning:
                return response.reasoning + '</think>' + response.text.strip() if response.text else response.reasoning, stop_reason
            elif hasattr(response, 'reasoning') and not response.reasoning:  # ealy exit of r1
                return response.text.strip(), stop_reason
            else:
                return response.text.strip(), stop_reason
        except Exception as e:
            if retry_count < 9:
                retry_count += 1
                print(f"Error during completion: {e}. Retrying ({retry_count})...")
                sleep(1)
            else:
                print("Max retries reached.")
                sleep(1)
                return "explicitly-supported", None
                    

def chat_with_retry(
    client: OpenAI,
    model_name,
    prompt,
    max_tokens,
    temperature=0.6,
    return_thinking=False,
):
    """retry whenever the request fails due to timeout
    the request will be sent by client.chat.completions.create
    
    Returns
    -------
    str, the response
    """
    retry_count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if return_thinking:
                thinking = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else response.choices[0].message.content.split('</think>')[0]
                return thinking, response.choices[0].message.content
            else:
                return response.choices[0].message.content
        except Exception as e:
            if retry_count < 9:
                retry_count += 1
                print(f"Error during chat completion: {e}. Retrying ({retry_count})...")
                sleep(retry_count) 
            else:
                print("Max retries reached.")
                return "", "explicitly-supported" if return_thinking else "explicitly-supported"


def do_zero_shot_predict(
    question, 
    answer, 
    sentence_ranges,
    base_url='http://localhost:8000/v1',
    model_name='/home/dingqiang/QwQ-32B/',
):
    """zero-shot method for fine-grained labeling

    Parameters
    ----------
    question : str
        question text
    answer : str
        answer text
    sentence_ranges : list
        list of sentence ranges, by default None


    Returns
    -------
    dict of list
        nli results, which has the following features:
        - nli_pred: list of str
        - note: list of dict, supplementary information for each prediction
    """
    client = OpenAI(
        base_url=base_url,  # 'https://openrouter.ai/api/v1',
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    nli_results = {
        'nli_pred': [],
        'note': []
    }
    if zero_shot_type == 'all_sentences':
        prompt = _build_zero_shot_prompt_all_sentences(question, answer, sentence_ranges)
        # prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        # print('debug: prompt', prompt)
        while True:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=10240,
                logprobs=False,  # logprobs will be used when </think> is in the response text
            ).choices[0].message.content
            if response.strip():
                break
            print("Warning: empty response, retrying...")

        print('debug: response', response)
        if re.search('```json\n([\s\S]+)\n```', response.split('</think>')[-1].strip()):
            data = json.loads(re.search('```json\n([\s\S]+)\n```', response.split('</think>')[-1].strip()).group(1))
        elif re.search('\[\n[\s\S]+\n\]', response.split('</think>')[-1].strip()):
            data = json.loads(re.search('\[\n[\s\S]+\n\]', response.split('</think>')[-1].strip()).group(0))
        else:
            raise ValueError(f"The response {response} does not contain valid JSON data. Please check the response format.")
        
        for i in range(len(sentence_ranges)):
            if i < len(data):
                nli_results['nli_pred'].append(idx2pred[pred2idx[data[i]['faithfulness'].lower()]])
                nli_results['note'].append(response if i == 0 else "")
            else:
                print(f"Warning: no prediction for sentence '{answer[sentence_ranges[i][0]:sentence_ranges[i][1]]}', using 'out-dependent' as default")
                nli_results['nli_pred'].append('out-dependent')
                nli_results['note'].append("")
    elif zero_shot_type == 'vanilla':
        for i, (a, b) in enumerate(sentence_ranges):
            sentence = answer[a:b]
            prompt = _build_zero_shot_prompt(question, answer, sentence)
            response = chat_with_retry(
                client,
                model_name,
                prompt,
                max_tokens=2048,
            )
            response = response.split('</think>')[-1].strip()
            nli_results['nli_pred'].append(get_classification(response))
            nli_results['note'].append(response)
    elif zero_shot_type == 'sequential_early_exit':
        tokenizer = AutoTokenizer.from_pretrained(openrouter2huggingface.get(model_name, None) or model_name)
        for i, (a, b) in enumerate(sentence_ranges):
            sentence = answer[a:b]
            prompt = _build_zero_shot_prompt(question, answer, sentence)
            # complete the response several times
            final_pred = 'explicitly-supported'  # default prediction
            final_note = ''
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            min_trial_num = 5
            i = 0
            class_probs = {
                'explicitly-supported': 0.0,
                'generally-supported': 0.0,
                'out-dependent': 0.0,
                'fabricated': 0.0,
                'inconsistent': 0.0,
            }
            while True:
                response = complete_with_retry(
                    client,
                    model_name,
                    prompt,
                    max_tokens=2048,
                    stop=['Wait', 'Alternatively'],
                    logprobs=True,  # logprobs will be used when </think> is in the response text
                )
                stop_reason = response.stop_reason if response.finish_reason == 'stop' and hasattr(response, 'stop_reason') else None
                i += 1
                if '</think>' in response.text.strip():  # the natural end of the response
                    note = response.text.strip()
                    pred, prob = get_classification_with_prob(response)
                    class_probs[pred] += prob
                    pred_idx = pred2idx[pred]
                    if pred_idx <= pred2idx[final_pred]:  # do not reserve the default prediction
                        final_pred = pred
                        final_note = json.dumps({"reasoning": note, "probs": class_probs})
                    # if pred2idx[final_pred] <= 1:
                    #     break
                    if i >= min_trial_num:
                        break
                    elif stop_reason in ['Wait', 'Alternatively']:
                        prompt += note.split('</think>')[0] + "\n\n" + stop_reason
                    else:
                        prompt += note.split('</think>')[0] + "\n\nWait"
                else:
                    prompt += response.text.strip()
                    # test the tendency of the LLM
                    trial = complete_with_retry(
                        client,
                        model_name,
                        prompt + "\n</think>\n",
                        max_tokens=16,
                        stop=['Wait', 'Alternatively'],
                        logprobs=True, 
                    )
                    note = response.text.strip() + '\n</think>\n' + trial.text.strip()
                    pred, prob = get_classification_with_prob(trial)
                    class_probs[pred] += prob
                    pred_idx = pred2idx[pred]
                    if pred_idx <= pred2idx[final_pred]:  # do not reserve the default prediction
                        final_pred = pred
                        final_note = json.dumps({"reasoning": note, "probs": class_probs})
                    # if pred2idx[final_pred] <= 1:  # temperarily disable the early exit
                    #     break
                    if stop_reason in ['Wait', 'Alternatively']:
                        prompt += "\n\n" + stop_reason
                    else:
                        prompt += "\n\nWait"  # to avoid the next round stop immediately

            nli_results['nli_pred'].append(final_pred)
            nli_results['note'].append(final_note)
            
    return nli_results


def get_classification(response: str):
    """find the label that appears most in the response;
    if multiple labels appear with the same frequency, return the label that appears most close to the beginning or the end of the response 
    """
    labels = [
        'explicitly-supported', 
        'generally-supported', 
        'inconsistent', 
        'fabricated', 
        'out-dependent',
        'ambiguous',
        'no-fact'
    ]
    response = response.lower()
    def get_label_position(label):
        start = end = 0
        positions = []
        while label in response[end:]:
            start = response.index(label, end)
            end = start + len(label)
            positions.append([start, end])
        return positions

    min_position = len(response)
    best_label = 'explicitly-supported'
    for label in labels:
        positions = get_label_position(label)
        current_min_position = min([min(a, len(response) - b) for a, b in positions]) if positions else len(response)
        if min_position > current_min_position:
            min_position = current_min_position
            best_label = label
    return best_label


def get_classification_with_prob(response: CompletionChoice):
    """find the label that appears most in the response

    Returns
    -------
    str
        the label that appears in the response (make sure the response is short enough to avoid multiple labels)
    float
        the geometric average of probs of the label tokens
    """
    labels = [
        'explicitly-supported', 
        'generally-supported', 
        'inconsistent', 
        'fabricated', 
        'out-dependent',
        'ambiguous',
        'no-fact'
    ]

    response_text = response.text.lower()
    start = response_text.index('</think>') + len('</think>') if '</think>' in response_text else 0
    text_offset = response.logprobs.text_offset
    for label in labels:
        if label in response_text[start:]:
            pred_start = response_text.index(label, start)
            pred_end = pred_start + len(label)
            # get the token indices
            token_indices = [i for i, offset in enumerate(text_offset) if offset+len(response.logprobs.tokens[i]) >= pred_start and offset < pred_end]
            # print('token_indices', token_indices)
            # print(pred_start, pred_end, label)
            assert token_indices
            # get the logprobs of the tokens
            logprobs = sum(response.logprobs.token_logprobs[token_indices[0]:token_indices[-1] + 1]) / len(token_indices)
            return label, math.exp(logprobs)
    print('Warning: no label found in the response:', response_text)
    return "explicitly-supported", 0.0