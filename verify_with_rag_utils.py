from zero_shot_utils import get_classification, chat_with_retry


def _build_prompt(document, retrieved_snippets, summary, sentence):
    retrieved_snippets = '\n\n'.join([f'Snippet {i+1}:\n{doc}' for i, doc in enumerate(retrieved_snippets)])
    return f"""You are judging whether a given sentence extracted from a summary is faithful to its source document. Several text snippets are retrieved to provide external world knowledge of the source document. The faithfulness has five options: 
A. Explicitly-Supported: all atomic facts of the sentence appear verbatim (up to a lexical or syntactic transformation) within the source document.
B. Generally-Supported: the source document entails the sentence, and the sentence is not explicitly-supported by the document. Note that some difference between part of the sentence and the document is allowed, but it should be restricted to the case in which the part of the sentence adopts a weaker or less certain utterance than the document. If any part of the sentence adopts a stronger or more certain utterance than the document, consider option D (Fabricated).
C. Inconsistent: the sentence logically contradicts the source document.
D. Fabricated: the sentence does not logically contradict the source document, and is neither logically implied by the source document nor by the retrieved snippets.
E. Out-Dependent: the sentence is not logically implied by the source document but by the union of the source document and the retrieved snippets.
F. Ambiguous: the sentence or the source document has multiple interpretations, which could lead to different conclusions.
G. No-Fact: the sentence is devoid of facts.

Your task is to decompose the target sentence into atomic facts and then output the correct option.

[Source Documents] {document}

[Retrieved Snippets] {retrieved_snippets}

[Summary] {summary}

[Sentence in the Summary] {sentence}

Your option:
"""


def do_verify_with_rag_predict(
    question,
    answer,
    sentences,
    client,
    nli_model,
    retriever,
):
    nli_results = {
        'nli_pred': [],
        'note': []
    }
    for s in sentences:
        # nli_pred = s['nli_pred']
        # if not nli_pred in ['fabricated', 'inconsistent']:
        #     nli_results['nli_pred'].append(nli_pred)
        #     nli_results['note'].append('Original prediction retained')
        #     continue
        retrieval_results = retriever.search(f"Provide more details about the claim: {s['text']}")
        thinking, response = chat_with_retry(
            client,
            nli_model,
            _build_prompt(
                    document=question,
                    retrieved_snippets=[r['contents'] for r in retrieval_results],
                    summary=answer,
                    sentence=s['text'],
                ),
            max_tokens=20480,
            return_thinking=True,
        )
        response = response.split('</think>')[-1].strip()
        classification = get_classification(response)
        # if get_classification(response) == 'out-dependent':
        #     classification = 'out-dependent'
        # else:
        #     classification = nli_pred
        nli_results['nli_pred'].append(classification)
        nli_results['note'].append({
            "retrieval_results": [r['contents'] for r in retrieval_results], 
            "response": f"{thinking}</think>\n{response}"
        })
    return nli_results