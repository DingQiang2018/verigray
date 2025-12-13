import sys
sys.path.append('/home/dingqiang/Infuse/')

from src.infuse import INFUSE


def do_infuse_predict(question, answer, sentence_ranges, model: INFUSE):
    """Infuse method for fine-grained labeling

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
    scores = model.process_document_summary(
        [question],
        [answer],
        1,
        1,
        summ_sentences=[[answer[start:end] for start, end in sentence_ranges]],
    )[0]
    results = {'nli_pred': ['entailment' if s > 0.5 else 'neutral' for s in scores], 'note': [""]*len(scores)}
    return results
