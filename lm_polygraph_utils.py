import json

import numpy as np
import torch
from lm_polygraph.utils.model import WhiteboxModel
# from lm_polygraph import estimate_uncertainty


def calc_greedy_probs(
    input_texts,
    output_texts,
    model: WhiteboxModel,
    n_alternatives=10,
):
    """to imitate the GreedyProbsCalculator.__call__(), without the generation of the output text

    Parameters
    ----------
    examples: dict of list with the following features:
        - question
        - answer
    model: WhiteboxModel

    Returns
    -------
    a dict of list with features:
        - `greedy_tokens_alternatives`
        - `greedy_tokens`
    """
    
    model.tokenizer.padding_side = "left"
    assert model.tokenizer.padding_side == "left"
    batch_input = model.tokenizer(input_texts, padding=True, return_tensors="pt")
    
    model.tokenizer.padding_side = "right"
    assert model.tokenizer.padding_side == "right"
    batch_output = model.tokenizer(output_texts, padding=True, return_tensors="pt")
    
    batch = {k: torch.cat([batch_input[k], batch_output[k]], dim=1).to(model.device()) for k in batch_input}
    with torch.no_grad():
        out = model(
            **batch,
            return_dict_in_generate=True,
        )
        logits = out.logits[:, batch_input['input_ids'].shape[1]-1:-1, :]

    cut_sequences = []
    cut_alternatives = []
    for i in range(len(input_texts)):
        seq = batch_output['input_ids'][i]
        length = len(seq)
        for j in range(len(seq)):
            if seq[j] == model.tokenizer.eos_token_id:
                length = j + 1
                break
        cut_sequences.append(seq[:length].tolist())
        cut_alternatives.append([[] for _ in range(length)])
        for j in range(length):
            lt = logits[i, j, :].cpu().numpy()
            best_tokens = np.argpartition(lt, -n_alternatives)
            ln = len(best_tokens)
            best_tokens = best_tokens[ln - n_alternatives : ln]
            for t in best_tokens:
                cut_alternatives[-1][j].append((t.item(), lt[t].item()))
            cut_alternatives[-1][j].sort(
                key=lambda x: x[0] == cut_sequences[-1][j],
                reverse=True,
            )

    result_dict = {
        "greedy_tokens": cut_sequences,
        "greedy_texts": [model.tokenizer.decode(seq) for seq in cut_sequences],
        "greedy_tokens_alternatives": cut_alternatives,
    }
    return result_dict


def do_ccp_predict(
    question,
    answer,
    sentence_ranges,
    ccp_model: WhiteboxModel,
    calc_claim_nli=None,
    calc_claim_extractor=None,
    estimator=None,
):
    """
    Use the LM-Polygraph lib to predict whether the answer is faithful to the question.
    Args:
        question: str, the question
        answer: str, the answer
        sentence_ranges: list of tuples, each tuple is (start, end) index of a sentence in the answer
        nli_model: WhiteBoxModel, the LM-Polygraph model
    """
    nli_results = {
        'nli_pred': [],
        'note': []
    }
    
    for a, b in sentence_ranges:
        texts = ccp_model.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"{question}\nSummarize the document above."},  # TODO: use the standard summarizer prompt
                {"role": "assistant", "content": answer[:b]}
            ],
            tokenize=False
        ).replace('</s>', '')
        input_texts = texts.split(answer[a:b])[0]
        output_texts = answer[a:b]

        # only for llama-2-7b-chat-hf
        if input_texts[-1] == ' ':
            input_texts = input_texts[:-1]
            output_texts = ' ' + output_texts

        deps = calc_greedy_probs([input_texts], [output_texts], model=ccp_model)
        deps.update({"greedy_texts" : [output_texts]})
        
        # extract claims according to sentence_ranges
        # `claims` is a list of list of Claim objects
        # Claim has attributes:
        # - claim_text: str
        # - sentence: str, the sentence from which the claim is extracted
        # - aligned_token_ids: list of int, Indices in the original generation of the tokens, which are related to the current claim
        deps.update(calc_claim_extractor(deps, texts=[input_texts], model=ccp_model))  # `texts` is only used to provide the batch size

        # calc_claim_nli input: `greedy_tokens_alternatives`
        # output: `greedy_tokens_alternatives_nli`
        deps.update(calc_claim_nli(deps, texts=None, model=ccp_model))
        
        # estimator input: 
        # - `claims`
        # - `greedy_tokens_alternatives`
        # - `greedy_tokens_alternatives_nli`
        uncertainty_scores = estimator(deps)[0]
        
        nli_pred = 'entailment' if all(us < -0.1 for us in uncertainty_scores) else 'neutral'
        note = json.dumps([{'claim_text': c.claim_text, 'uncertainty_score': s} for c, s in zip(deps['claims'][0], uncertainty_scores)])
        nli_results['nli_pred'].append(nli_pred)
        nli_results['note'].append(note)

    return nli_results