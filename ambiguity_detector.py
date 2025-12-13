import os
from argparse import ArgumentParser

import numpy as np
from openai import OpenAI
from datasets import load_dataset, Dataset
# from sklearn.metrics import confusion_matrix

from zero_shot_utils import chat_with_retry


LLM_judges = [
    {'name': 'openai/o3-mini', 'temperature': None},
    {'name': 'openai/gpt-4o', 'temperature': 1},
    {'name': 'google/gemini-2.0-flash-001', 'temperature': 1},
    {'name': 'meta-llama/llama-3.1-405b-instruct', 'temperature': 1},
    {'name': 'openai/gpt-5', 'temperature': None}
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('OPENROUTER_API_KEY')
)


def fact_checking(document, statement, llm):
    prompt_template = """Instructions:

1. You have been given a STATEMENT and some DOCUMENT.

2. Determine whether the given STATEMENT is supported by the given DOCUMENT. The STATEMENT does not need to be explicitly supported by the DOCUMENT but should be strongly implied by the DOCUMENT.

3. Before showing your answer, think step-by-step and show your specific reasoning. As part of your reasoning, summarize the main points of the DOCUMENT.

4. If the STATEMENT is supported by the DOCUMENT, be sure to show the supporting evidence. 

5. After stating your reasoning, restate the STATEMENT and then determine your final answer based on your reasoning and the STATEMENT.

6. Your final answer should be either [Attributable] or [Not Attributable], or [Contradictory].

7. Wrap your final answer in square brackets.

DOCUMENT:

{document}

STATEMENT:

{statement}
"""
    response = chat_with_retry(
        client,
        llm['name'],
        prompt=prompt_template.format(document=document, statement=statement),
        max_tokens=10240,
        temperature=llm['temperature']
    )
    print('Fact Checking Response:', response)
    print('\n', '='*80,'\n')
    return response


def evaluate_completeness(document, statement, response, llm):
    prompt_template = """You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.

### Task Description
You ar evaluating whether the model (a fact verifier)'s response explicitly verifies all critical components of the given statement, regardless of verification accuracy.

The output format should be as follows:

Feedback: (provide a detailed evaluation based on completeness) [RESULT] (an integer, either 0 or 1)

Do not include any additional opening statements, explanations, or conclusions beyond the specified format.

### Evaluation Policy
Before assigning a score, evaluate the response by checking:
1. Thoroughness of Verification
2. Completeness of Checking
3. Avoidance of Partial Verification
4. Strict Relevance

If any of these checks fail, the response should be rated 0.

Document:
{document}

Statement:
{statement}

Model Response:
{response}

### Score Rubric (Completeness Evaluation)

Score 0 (Low Completeness): The response does notverify all aspects of the statement. It either skips key parts, provides an incomplete assessment, or focuses on only a subset of the facts presented in the statement. The response is not a thorough verification and cannot be considered fully precise.

Score 1 (High Completeness): The response explicitly checks every aspect of the statement. Each part of the statement is verified without omission, and no essential details are overlooked. The response ensures that the verification is complete, without introducing unrelated details.

Please review the model response above carefully. Then, output your evaluation in the following format:
Feedback: <detailed feedback explaining your evaluation> [RESULT] <a score between 0 (poor) and 1 (excellent)>
"""
    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model=llm['name'],
                messages=[{"role": 'user', 'content': prompt_template.format(document=document, statement=statement, response=response)}],
                temperature=llm['temperature'],
                max_tokens=10240
            ).choices[0].message.content
            print('Completeness Response:', response)
            print('\n', '='*80,'\n')

            score = int(response.split('[RESULT]')[-1].strip()[0])
            break
        except Exception as e:
            print("Error in evaluating completeness, retrying...", e)
            continue
    return score == 1 


def evaluate_logic_coherence(document, statement, response, llm):
    prompt_template = """You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.

### Task Description
You are evaluating whether the model (a fact verifier)'s reasoning logically aligns with its final verdict, regardless of the correctness of individual inference steps.

The output format should be as follows:

Feedback: (provide a detailed evaluation based on logical coherency) [RESULT] (an integer, either 0 or 1)

Do not include any additional opening statements, explanations, or conclusions beyond the specified format.

### Evaluation Policy
Before assigning a score, evaluate the response by checking: 1. Logical Consistency with the Label
2. Avoidance of Arbitrary Conclusions
3. Internal Coherence
4. Alignment with the Given Evidence

If any of these checks fail, the response should be rated 0.

Document:
{documents}

Statement:
{statement}

Model Response:
{response}

### Score Rubric (Logical Coherency Evaluation)

Score 0 (Low Logical Coherency): The model's reasoning does not logically support the assigned label. The reasoning process either contradicts the label, includes an arbitrary conclusion, lacks logical progression, or misapplies the classification rules. The model mayhave arrived at the correct label by luck rather than through valid reasoning, or its explanation contains gaps that prevent a clear justification of the assigned label. If the reasoning indicates one label but the response assigns a different label, or if the reasoning is internally inconsistent, the response must be rated 0.

Score 1 (High Logical Coherency): The model's reasoning process is fully aligned with the assigned label, with each step logically supporting the final classification. The reasoning follows a structured and internally consistent path without contradictions, arbitrary jumps, or gaps. The model applies the labeling criteria correctly, ensuring that the conclusion is reached through a valid and justified reasoning process rather than by chance. 

Please review the model response above carefully. Then, output your evaluation in the following format:
Feedback: <detailed feedback explaining your evaluation> [RESULT] <a score between 0 (poor) and 1 (excellent)>
"""
    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model=llm['name'],
                messages=[{"role": 'user', 'content': prompt_template.format(documents=document, statement=statement, response=response)}],
                temperature=llm['temperature'],
                max_tokens=10240
            ).choices[0].message.content
            print('Logic Coherence Response:', response)
            print('\n', '='*80,'\n')

            score = int(response.split('[RESULT]')[-1].strip()[0])
            break
        except Exception as e:
            print("Error in evaluating logic coherence, retrying...", e)
            continue
    return score == 1


def evaluate_reasoning_quality(document, statement, response, llm):
    prompt_template = """You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.

### Task Description
You are evaluating whether each inference step in the model (a fact verifier)'s reasoning is logically sound and justified. Specifically, we evaluate the internal consistency and validity of rationales.

The output format should be as follows:

Feedback: (provide a detailed evaluation based on reasoning quality) [RESULT] (an integer, either 0 or 1)
 
Do not include any additional opening statements, explanations, or conclusions beyond the specified format.
 
### Evaluation Policy
Before assigning a score, evaluate the response by checking:
1. Accuracy of Logical Steps
2. Avoidance of Incorrect Inferences
3. Consistency in Fact Interpretation
4. Sound Justification for Verification

If any of these checks fail, the response should be rated 0.

Document:
{document}

Statement:
{statement}

Model Response:
{response}

### Score Rubric (Reasoning Quality Evaluation)

Score 0 (Low Reasoning Quality): The model's reasoning is flawed, misinterprets the document, or applies incorrect logic when verifying the statement. The response contains unsupported inferences, misreads facts, distorts meaning, or jumps to a conclusion that is not justified by the evidence. The model may misapply classification rules, make unwarranted logical assumptions, or draw causality that is not present in the document. If the response demonstrates any fundamental flaws in reasoning, the score must be 0.

Score 1 (High Reasoning Quality): The model accurately applies logical reasoning and correctly interprets the document when verifying the statement. The reasoning properly connects facts, avoids misinterpretations, and maintains logical validity from extraction to conclusion. The model does not introduce unjustified assumptions or incorrectly infer relationships between facts, and it provides a structured, step-by-step justification that correctly supports its classification.

Please review the model response above carefully. Then, output your evaluation in the following format:
Feedback: <detailed feedback explaining your evaluation> [RESULT] <a score between 0 (poor) and 1 (excellent)>
"""
    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model=llm['name'],
                messages=[{'role': 'user', 'content': prompt_template.format(document=document, statement=statement, response=response)}],
                temperature=llm['temperature'],
                max_tokens=10240
            ).choices[0].message.content
            print('Reasoning Quality Response:', response)
            print('\n', '='*80,'\n')

            score = int(response.split('[RESULT]')[-1].strip()[0])
            break
        except Exception as e:
            print("Error in evaluating reasoning quality, retrying...", e)
            continue
    return score == 1


def agreed_by_others(document, statement, fc_result, llm):
    others = [x for x in LLM_judges if x != llm]
    if not evaluate_completeness(document, statement, fc_result, others[0]):
        return False
    if not evaluate_logic_coherence(document, statement, fc_result, others[1]):
        return False
    if not evaluate_reasoning_quality(document, statement, fc_result, others[2]):
        return False
    return True


def pred_ambiguity(document, statement, original_label):
    if not original_label.lower() in ['explicitly-supported', 'implicitly-supported', 'fabricated', 'inconsistent', 'contradicting']:
        return False, ""
    
    history = ""
    for llm in LLM_judges:
        fc_result = fact_checking(document, statement, llm)
        if ('[Not Attributable]' in fc_result or '[Contradictory]' in fc_result) and 'supported' in original_label.lower():
            # if agreed_by_others(document, statement, fc_result, llm):
                return True, f"{llm}: {fc_result}"
        if '[Attributable]' in fc_result and original_label.lower() in ['fabricated', 'inconsistent', 'contradicting']:
            # if agreed_by_others(document, statement, fc_result, llm):
                return True, f"{llm}: {fc_result}"
        history += f"{llm}: {fc_result}\n{"-"*20}\n"

    return False, history


def worst_label_original(sentences):
    worst =''
    for s in sentences:
        if 'unwanted' in s['annotation'].lower():
            return 'unwanted'
        elif s['annotation']:
            worst = s['annotation']
    return worst


def worst_label_current(sentences):
    label2score = {
        'explicitly-supported': 6,
        'implicitly-supported': 5,
        'generally-supported': 5,
        'out-dependent': 4,
        'ambiguous': 3,
        'fabricated': 2,
        'inconsistent': 1,
        'no-fact': 6,
    }
    worst = 'explicitly-supported'
    for s in sentences:
        if label2score[s['annotation'].lower()] < label2score[worst]:
            worst = s['annotation'].lower()
    return worst


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-file', type=str, default='/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_20250916.jsonl')
    parser.add_argument('--n-samples', type=int)
    args = parser.parse_args()

    if os.path.isfile('ambiguity_results/results.jsonl'):
        print("Found existing results, loading...")
        results = load_dataset('json', data_files='ambiguity_results/results.jsonl')['train']
        ambiguity_pred = results['ambiguity_pred']
    else:
        ds = load_dataset('json', data_files=args.data_file)['train']
        if args.n_samples < len(ds):
            ds = ds.select(range(args.n_samples+250, len(ds)))

        ambiguity_pred = []
        ambiguity_rationale = []
        documents = ds['question']
        summaries = ds['answer']
        sentences = ds['sentences']
        new_sentences = []
        for i in range(len(ds)):
            print(i+1, '/', len(ds))
            print('Document:', documents[i])
            print('Summary:', summaries[i])
            sentencewise_pred = []
            for j in range(len(sentences[i])):
                print(f"  Sentence {j+1}: {sentences[i][j]['text']}")
                print(f"    Original Label: {sentences[i][j]['annotation']}")
                pred, rationale = pred_ambiguity(documents[i], sentences[i][j]['text'], sentences[i][j]['annotation'])
                sentencewise_pred.append(
                    {
                        'text': sentences[i][j]['text'],
                        'annotation': sentences[i][j]['annotation'],
                        'ambiguity_pred': pred,
                        'ambiguity_rationale': rationale
                    }
                )
            new_sentences.append(sentencewise_pred)
        # save results
        result_ds = Dataset.from_dict({
            'question': documents,
            'answer': summaries,
            'sentences': new_sentences,
        })
        os.makedirs('ambiguity_results', exist_ok=True)
        result_ds.to_json('ambiguity_results/results.jsonl')
