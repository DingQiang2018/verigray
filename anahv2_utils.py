import re

import openai
from openai import OpenAI


def hallucination_check_prompt(question, reference, annotation, language):
    cn_user_prompt = f"""
    你将作为一个‘幻觉’标注器，我将会给你提供一个一个问题，一个针对该问题的部分回答和相关的参考要点。你需要判断提供的回答中是否含有幻觉性内容，并标注幻觉类型。

    ‘幻觉’指的是与参考要点相矛盾或在参考要点中没有依据的内容。

    ## 判断准则：

    1. **无幻觉：** 如果回答与参考要点完全一致，且没有引入与参考要点相矛盾的信息，请输出：<无幻觉>。
    2. **矛盾：** 如果回答内容与参考要点存在明显矛盾，请输出：<矛盾>。
    3. **无法验证：** 如果回答包含的信息在参考要点中没有提及，且无法从参考要点中得到支持或验证，请输出：<无法验证>。

    ## 任务流程：

    ### 1. **仔细阅读问题，问题如下：** 

    {question}

    ### 2. **仔细阅读回答，部分回答如下：** 

    {annotation}

    ### 3. **仔细阅读参考要点，参考要点如下：**

    {reference} 

    ### 4. **进行分析：** 根据上述判断标准，判断回答中是否包含幻觉，并输出幻觉类型。
    """

    en_user_prompt = f"""
    You will act as a 'Hallucination' annotator. I will provide you with a question, a partial answer to that question, and related reference points. You need to determine whether the provided answer contains any hallucinatory content and annotate the type of hallucination.

    'Hallucination' refers to content that contradicts the reference points or is unsupported by them.

    ## Judgment Criteria:

    1. **No Hallucination:** If the answer is completely consistent with the reference points and does not introduce any contradictory information, output: <No Hallucination>.
    2. **Contradiction:** If the answer clearly contradicts the reference points, output: <Contradictory>.
    3. **Unverifiable:** If the answer contains information not mentioned in the reference points and cannot be supported or verified by them, output: <Unverifiable>.

    ## Task Process:

    ### 1. **Carefully read the question, which is as follows:** 

    {question}

    ### 2. **Carefully read the partial answer, which is as follows:** 

    {annotation}

    ### 3. **Carefully read the reference points, which are as follows:**

    {reference} 

    ### 4. **Conduct the analysis:** Based on the above judgment criteria, determine if the answer contains hallucinations and output the type of hallucination.
    """

    return cn_user_prompt if language == "zh" else en_user_prompt


def do_anahv2_predict(
    prompt: str, 
    response: str, 
    query: str,
    document: str,
    sentence_ranges, 
    base_url, 
    nli_model
):
    if not document:
        document = prompt
        query = "According to the document, what can we infer?"

    client = OpenAI(api_key="none", base_url=base_url)
    nli_results = {
        'nli_pred': [],
        'note': []
    }

    for (start, end) in sentence_ranges:
        claim = response[start:end]
        p = hallucination_check_prompt(query, document, claim, "en")
        try:
            r = client.chat.completions.create(
                model=nli_model,
                messages=[
                    {'role': 'user', 'content': p},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            r = r.choices[0].message.content
        except openai.BadRequestError as e:
            print(e)
            r = "<No Hallucination>\nopenai.BadRequestError"
        if re.match('<No Hallucination>', r):
            nli_results['nli_pred'].append('entailment')
        elif re.match('<Contradictory>', r):
            nli_results['nli_pred'].append('contradiction')
        elif re.match('<Unverifiable>', r):
            nli_results['nli_pred'].append('neutral')
        else:
            print('Warning: the prediction is', r)
            nli_results['nli_pred'].append('entailment')
        nli_results['note'].append(r)
    return nli_results