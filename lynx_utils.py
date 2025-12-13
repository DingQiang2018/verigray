from openai import OpenAI


lynx_prompt_template = """Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}"""


def do_lynx_predict(
    prompt: str,
    response: str,
    query: str,
    document: str,
    sentence_ranges: list,
    base_url: str = 'http://localhost:8000/v1',
    nli_model: str = '/home/dingqiang/lynx-8b-v1.1',
):
    """call the Lynx model through openAI API to predict the entailment of each sentence in the answer

    Parameters
    ----------
    question : str
        the document
    answer : str
        the summary
    sentence_ranges: list
        the ranges of each sentence in the answer

    Returns
    -------
    nli_results: dict of list, which has the following features:

    - nli_pred: list of str
        the predicted entailment of each sentence
    - note: list of str
        supplementary info for each sentence
    """
    if not document:
        document = prompt
        query = "According to the document, what can we infer?"

    client = OpenAI(api_key="none", base_url=base_url)
    nli_results = {
        'nli_pred': [],
        'note': []
    }
    for i in range(len(sentence_ranges)):
        claim = response[sentence_ranges[i][0]:sentence_ranges[i][1]]
        p = lynx_prompt_template.format(context=document, question=query, answer=claim)
        r = client.chat.completions.create(
            model=nli_model,
            messages=[
                {'role': 'user', 'content': p},
            ],
            max_tokens=256,
            temperature=0
        )
        r = r.choices[0].message.content
        if 'PASS' in r:
            nli_results['nli_pred'].append('entailment')
        elif 'FAIL' in r:
            nli_results['nli_pred'].append('neutral')
        else:
            print('Warning: the prediction is not PASS or FAIL, but:', r)
            nli_results['nli_pred'].append('entailment')
        nli_results['note'].append(r)
    return nli_results
    
