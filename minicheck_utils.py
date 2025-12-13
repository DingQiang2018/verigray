from minicheck.minicheck import MiniCheck


def do_minicheck_predict(
    scorer: MiniCheck,
    prompt: str,
    response: str,
    query: str,
    document: str,
    sentence_ranges: list
):
    """use minicheck to check if the document in the question entails the answer

    Parameters
    ----------
    scorer : MiniCheck
        minicheck scorer
    question : str
    answer : str
    sentence_ranges : list
        sentence ranges of the answer
    """
    if not document:
        document = prompt

    nli_results = {
        'nli_pred': [],
        'note': []
    }
    claims = [response[a:b] for (a,b) in sentence_ranges]
    preds, probs, _, _ = scorer.score(docs=[document]*len(claims), claims=claims)
    for i in range(len(sentence_ranges)):
        if preds[i] == 1:
            nli_results['nli_pred'].append('entailment')
        elif preds[i] == 0:
            nli_results['nli_pred'].append('neutral')
        else:
            raise ValueError(f'pred_label {preds[0]} not supported')
        nli_results['note'].append(probs[i])
    return nli_results