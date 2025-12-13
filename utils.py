import json

from openai import OpenAI
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


prompt_template = """DECONTEXTUALIZATION CRITERIA: Decontextualization adds the right type of information to a CLAIM to make it standalone. This process can modify the original CLAIM in the following manners: 
- Substituting pronouns or incomplete names with the specific subject being referred to. 
- Including contextual information to provide more context about the subject. 

Instructions:- Identify the "subject" of the claim and locate the claim within the context. 
- Use the CONTEXT to substitute any incomplete names or pronouns in the CLAIM. 
- If there is no decontextualization necessary, return the original claim as is. 
- The decontextualization should minimally modify the claim by only adding necessary contextual information. 
- Refer to the following examples to understand the task and output formats. 

Example 1: 
CONTEXT: Almondbury Community School bullying incident: The clip shows the victim, with his arm in a cast, being dragged to the floor by his neck as his attacker says "I'll drown you" on a school playing field, while forcing water from a bottle into the victim's mouth, simulating waterboarding. The video was filmed in a lunch break. The clip shows the victim walking away, without reacting, as the attacker and others can be heard continuing to verbally abuse him. The victim, a Syrian refugee, had previously suffered a broken wrist; this had also been investigated by the police, who had interviewed three youths but took no further action. 
CLAIM: The victim had previously suffered a broken wrist. 
DECONTEXTUALIZED CLAIM: The Syrian refugee victim in the Almondbury Community School bullying incident had previously suffered a broken wrist. 

Example 2: 
CONTEXT: Isaiah Stewart: Stewart was born in Rochester, New York. He grew up playing soccer and boxing. 
CLAIM: He grew up playing boxing. 
DECONTEXTUALIZED CLAIM: Isaiah Stewart grew up playing boxing. 

Example 3: 
CONTEXT: Arab Serai: According to S.A.A. Naqvi, Mughal emperor Humayun's widow Haji Begum built this "serai" in c. 1560/61 to shelter three hundred Arab mullahs whom she was taking with her during her "hajj" to Mecca; however, Y.D. Sharma opines that the word Arab in the title is a misnomer as this building was built for the Persian craftsmen and workers who built the Humayun's Tomb. In January 2017, the Aga Khan Trust for Culture started a project to conserve the "serai". The restoration was completed in November 2018. In March 2019, the trust announced a planned project to conserve the "baoli" (stepwell) of the serai with the help of funds from the embassy of Germany. 
CLAIM: The planned project is to conserve the "baoli" (stepwell) of the serai. 
DECONTEXTUALIZED CLAIM: The Aga Khan Trust for Culture's planned project in March 2019 is to conserve the "baoli" (stepwell) of the Arab Serai. 

Example 4: 
CONTEXT: Mason Warren: Warren was born in Doncaster, South Yorkshire and started his career with Rotherham United, where he progressed from the youth team to sign a professional contract in May 2015. He was taken with the first team on the pre-season tour of Scotland and became a regular with the development squad before he was sent to NPL Division One South side Sheffield on a two-month youth loan deal. He was a prominent figure in the side making six appearances during his loan spell before he was recalled in early January 2016. In February 2016, he was loaned out again joining National League North side Harrogate Town on a one-month loan deal. After picking up the Player of the Month award for Harrogate during February, his loan was extended until April. He went on to make a total of eleven appearances for Town. Upon his return to Rotherham in April, he signed a new two-year contract extension until 2018. 
CLAIM: He signed a new two-year contract extension until 2018. 
DECONTEXTUALIZED CLAIM: Mason Warren Warren signed a new two-year contract extension until 2018 with Rotherham United. 

Example 5: 
CONTEXT: Lost Girls (band): Lost Girls is a band that primarily consists of Patrick Fitzgerald and Heidi Berry. They formed in 1998 after Fitzgerald left Kitchens of Distinction and Berry left 4AD, which had released three of her albums after her appearance on This Mortal Coil's 1991 album "Blood". 
CLAIM: 4AD had released three of her albums. 
DECONTEXTUALIZED CLAIM: 4AD had released three of Heidi Berry's albums before she left to form Lost Girls. 

Example 6: 
CONTEXT: Bernard Joseph (politician): He was a member of the Congress of the People before he joined the Economic Freedom Fighters. Joseph said that he left the party because he felt that the party lacked leadership and movement. He joined the Economic Freedom Fighters to implement the party's policies. 
CLAIM: He joined the party to implement the party's policies. 
DECONTEXTUALIZED CLAIM: Bernard Joseph joined the Economic Freedom Fighters to implement the party's policies. 

Example 7: 
CONTEXT: Ham Sandwich (song): On February 20, 2019, the song was self-released as a digital download on international digital stores, as well as being released through various music streaming services. The song was released partially as a response to fans who were displeased with Getter's album "Visceral", released in late 2018. It was also released shortly before the launch of his "Visceral Tour", based off of his album of the same name. 
CLAIM: The album and tour are both named "Visceral". 
DECONTEXTUALIZED CLAIM: Getter's album and tour are both named "Visceral". 

Similarly, generate a decontextualized claim for the following pair of CLAIM and CONTEXT making minimal alterations to the original structure of the CLAIM while ensuring clarity and coherence. 

CONTEXT: {context}
CLAIM: {sentence}
DECONTEXTUALIZED CLAIM: """


def chat_openai(
    prompt, 
    base_url="http://localhost:8000/v1", 
    api_key="none",
    model="/home/dingqiang/Qwen2-7B-Instruct", 
):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


def decontextualize(
    # question: str, 
    text: str, 
    sentences, 
    openai_args
):
    """decontextualize sentences in text by prompting an LLM, where the
    context may have a prefix that is not part of the sentences

    Parameters
    ----------
    text: str
    sentences : list of dict, which has the following features:
        text: key with the sentence text
        range: key with the start and end index of the sentence in the context
    openai_args : dict
        arguments for the openai chat API
    
    Returns
    -------
    new_text : str
        the new text with the decontextualized sentences
    new_sentences : list of dict, which has the following features:
        text: key with the decontextualized sentence text
        range: key with the start and end index of the sentence in the new context
    """
    new_text = ''
    # sentences = nltk.sent_tokenize(text)
    # sentences = [{'text': s, 'range': [text.index(s), text.index(s)+len(s)]} for s in sentences]

    new_sentences = []
    for i in range(len(sentences)):
        prompt = prompt_template.format(context=text, sentence=sentences[i]['text'])
        decontextualized_sentence = chat_openai(prompt, **openai_args)
        new_sentences.append({
            'text': decontextualized_sentence,
            'range': [len(new_text), len(new_text)+len(decontextualized_sentence)],
            **{k: v for k, v in sentences[i].items() if k not in ['text', 'range']}
        })
        new_text += decontextualized_sentence+' ' if decontextualized_sentence[-1] != ' ' else decontextualized_sentence
    return new_text, new_sentences


def evaluate(ds, uncertainty_thresh=-0.001):
    """evalute the performance of the hallucination detection predictions in nli_pred

    Parameters
    ----------
    ds : dataset
        the dataset that has the following features

        - question: str, the question
        - answer: str, the answer
        - sentences: list of dict, which has the following features:
            - text: key with the sentence text
            - range: key with the start and end index of the sentence in the context
            - annotation: key with the annotation of the sentence
            - nli_pred: key with the predicted entailment of the sentence
            - nli_note: key with the supplementary info for the sentence
            - rationale
            - note
        
    Returns
    -------
    None
    """
    # calculating the accuracy and macro F1 score
    pred2num = {'entailment': 2, 'neutral': 1, 'contradiction': 0}

    # only for redeep
    topk_head_path = '/home/dingqiang/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/topk_heads.json'
    beta = 1.6
    head_topk = 7
    layer_topk = 3
    with open(topk_head_path,'r') as f:
        # [(layer, head)...]
        copy_heads = json.load(f)[:head_topk]

    preds = []
    labels = []
    sentences = ds['sentences']
    for i in range(len(ds)):
        for j in range(len(sentences[i])):
            if sentences[i][j]['annotation'].lower() != 'no-fact':
                try:
                    nli_note = json.loads(sentences[i][j]['nli_note']) if type(sentences[i][j]['nli_note']) == str else sentences[i][j]['nli_note']
                    if type(nli_note) == list and nli_note and nli_note[0].get('uncertainty_score', None) is not None:  # for ccp and LLM-Check
                        uncertainty_score = max([x['uncertainty_score'] for x in nli_note])
                        pred = 'entailment' if uncertainty_score < uncertainty_thresh else 'neutral' 
                    elif type(nli_note) == dict and nli_note.get('parameter_knowledge_score', None) is not None:  # for ReDeEP
                        parameter_knowledge_scores = nli_note['parameter_knowledge_score']
                        prompt_attention_scores = nli_note['prompt_attention_score']
                        parameter_score = np.mean([parameter_knowledge_scores['layer_'+str(layer)] for layer in range(32-layer_topk,32)])
                        attention_score = np.mean([prompt_attention_scores[f'({h[0]}, {h[1]})'] for h in copy_heads])
                        uncertainty_score = parameter_score - beta*attention_score
                        pred = 'entailment' if uncertainty_score < uncertainty_thresh else 'neutral'
                    else:
                        pred = sentences[i][j]['nli_pred']
                except:
                    pred = sentences[i][j]['nli_pred']
                preds.append(pred2num[pred.lower()])
                labels.append(fine_grained_pred2num(sentences[i][j]['annotation'].lower()))  # ambiguous if not found

    ## do calculation
    # acc = np.sum(preds == labels)/len(preds)
    balanced_acc, h_r, h_p, h_f1 = regular_evaluation(preds, labels)

    rank_loss = get_rank_loss([p >= 2 for p in preds], labels)
    print(f'balanced acc={balanced_acc}, hallucination recall={h_r}, hallucination precision={h_p}, hallucination F1={h_f1}, rank loss={rank_loss}')


def get_rank_loss(preds, labels):
    """calculate the ranking loss
    """
    P_size = 0
    rank_loss = 0.0
    for i in range(len(labels)-1):
        for j in range(i+1, len(labels)):
            if labels[i] > labels[j]:
                P_size += 1
                if preds[i] < preds[j]:
                    rank_loss += 1
                elif preds[i] == preds[j]:
                    rank_loss += 0.5
            elif labels[i] < labels[j]:
                P_size += 1
                if preds[i] > preds[j]:
                    rank_loss += 1
                elif preds[i] == preds[j]:
                    rank_loss += 0.5
    if P_size > 0:
        return rank_loss/P_size
    else:
        return 0.0


def regular_evaluation(preds, labels):
    """filter out not sure instances and calculate bacc and hallucination F1
    """
    preds = np.array(preds)
    labels = np.array(labels)

    fabricated = fine_grained_pred2num('fabricated')
    implicitly_supported = fine_grained_pred2num('implicitly-supported')
    preds = preds[np.logical_or(labels <=fabricated, labels >=implicitly_supported)] <= 1
    labels = labels[np.logical_or(labels <=fabricated, labels >=implicitly_supported)] <= fabricated
    balanced_acc = 0.5*(np.sum(np.logical_and(preds == labels, labels == 0))/np.sum(labels == 0)\
                    + np.sum(np.logical_and(preds == labels, labels == 1))/np.sum(labels == 1))
    
    confusion = confusion_matrix(labels, preds)
    h_r = confusion[1,1]/np.sum(confusion[1])
    h_p = confusion[1,1]/np.sum(confusion[:, 1])
    h_f1 = 2*h_r*h_p/(h_r+h_p)

    return balanced_acc, h_r, h_p, h_f1


def fine_grained_pred2num(pred: str):
    pred_to_num = {
        'explicitly-supported': 6, 
        'generally-supported': 5, 
        'implicitly-supported': 5, 
        'out-dependent': 4,
        'no-fact': 3,
        'ambiguous': 2,
        'fabricated': 1, 
        'inconsistent': 0,
        'contradicting': 0,
    }
    if pred in pred_to_num:
        return pred_to_num[pred.lower()]
    else:
        return 3
    

def fine_grained_evaluate(ds, thresh=5, print_results=True):
    """evaluate the performance of the hallucination detection predictions in nli_pred
    Parameters
    ----------
    ds : dataset
        the dataset that has the following features
        - question: str, the question
        - answer: str, the answer
        - sentences: list of dict, which has the following features:
            - text: key with the sentence text
            - range: key with the start and end index of the sentence in the context
            - annotation: key with the annotation of the sentence
            - nli_pred: key with the predicted entailment of the sentence
            - nli_note: key with the supplementary nli info for the sentence
            - rationale
            - note
    thresh : int
        the threshold to separate hallucination and non-hallucination
        thresh = 4 means implicitly-supported and above are non-hallucination
    
    Returns
    -------
    dict, with the following keys
        - balanced_acc: float, the balanced accuracy
        - hallucination_recall: float, the hallucination recall
        - hallucination_precision: float, the hallucination precision
        - hallucination_f1: float, the hallucination F1 score
        - rank_loss: float, the ranking loss
        - macro_f1: float, the macro F1 score
    """
    preds = []
    labels = []
    sentences = ds['sentences']
    for i in range(len(ds)):
        for j in range(len(sentences[i])):
            preds.append(fine_grained_pred2num(sentences[i][j]['nli_pred'].lower()))
            labels.append(fine_grained_pred2num(sentences[i][j]['annotation'].lower()))
    preds = np.array(preds)
    labels = np.array(labels)

    confusion = confusion_matrix(labels, preds, labels=list(range(7)))
    if print_results:
        print('confusion matrix:', confusion)

    hh = confusion[:2,:thresh].sum()
    hf = confusion[:2,thresh:].sum()
    fh = confusion[5:,:thresh].sum()
    ff = confusion[5:,thresh:].sum()
    balanced_acc = 0.5*hh/(hf+hh) + 0.5*ff/(ff+fh)
    if print_results:
        print('balanced accuracy:', balanced_acc)

    p = hh/(hh+fh)
    r = hh/(hh+hf)
    if print_results:
        print('hallucination recall:', r)
        print('hallucination precision:', p)
        print('hallucination detection f1:', 2*p*r/(p+r))

    rank_loss = get_rank_loss([p >= 5 for p in preds], labels)
    if print_results:
        print('rank loss:', rank_loss)

    macro_f1 = f1_score(labels, preds, average='macro')
    if print_results:
        print('macro f1:', macro_f1)
    
    return {
        'confusion_matrix': confusion,
        'balanced_acc': balanced_acc,
        'hallucination_recall': r,
        'hallucination_precision': p,
        'hallucination_f1': 2*p*r/(p+r),
        'rank_loss': rank_loss,
        'macro_f1': macro_f1,
    }
