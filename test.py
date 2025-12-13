import os
import sys
sys.path.append('/home/dingqiang/')

from datasets import load_dataset
from argparse import ArgumentParser

from utils import evaluate, fine_grained_evaluate
from lynx_utils import do_lynx_predict
from anahv2_utils import do_anahv2_predict
from zero_shot_utils import do_zero_shot_predict
from verify_with_rag_utils import do_verify_with_rag_predict


parser = ArgumentParser()
parser.add_argument('--method', type=str, choices=['lynx', 'minicheck', 'anahv2', 'zero_shot', 'infuse', 'ccp', 'verify_with_rag', 'redeep', 'llm_check'])
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--data-file', type=str, default=None)
parser.add_argument('--cache-file', type=str, default=None)
parser.add_argument('--dataset', type=str, default='', choices=['verigray'])
parser.add_argument('--n-samples', type=int, default=None)
# NLI
parser.add_argument('--base-url', type=str, default='https://openrouter.ai/api/v1')
parser.add_argument('--model-name', type=str, default='')
parser.add_argument('--uncertainty-thresh', type=float, default=None)
parser.add_argument('--fine-grained-annotation', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    data_files = {
        'verigray': '/home/dingqiang/hallucination_detection/FaithBench_sentence_level_reannotated_20251007.jsonl',
    }
    data_file = args.data_file if args.data_file else data_files[args.dataset]
    cache_file = args.cache_file if args.cache_file else f'{args.method}_results/{args.dataset}_sentence_level_results.jsonl'  # only used for reading

    if os.path.exists(cache_file) and not args.overwrite:
        print('Loading the results from file')
        ds = load_dataset('json', data_files=cache_file)['train']

        if args.n_samples is not None and len(ds) > args.n_samples:
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(args.n_samples))
    else:
        ds = load_dataset('json', data_files=data_file)['train']
        if args.n_samples is not None and len(ds) > args.n_samples:
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(args.n_samples))

        if args.method == 'minicheck':
            from minicheck_utils import do_minicheck_predict
            from minicheck.minicheck import MiniCheck
            minicheck_scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=True, max_model_len=8192)
        elif args.method == 'infuse':
            from infuse_utils import do_infuse_predict, INFUSE
            infuse_model = INFUSE(args.model_name if args.model_name else 'cross-encoder/nli-deberta-v3-large')
        elif args.method == 'ccp':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from lm_polygraph.utils.model import WhiteboxModel
            from lm_polygraph.utils.deberta import Deberta
            from lm_polygraph.utils.openai_chat import OpenAIChat
            from lm_polygraph.stat_calculators import GreedyAlternativesNLICalculator, ClaimsExtractor
            from lm_polygraph.estimators import ClaimConditionedProbabilityClaim

            from lm_polygraph_utils import do_ccp_predict

            assert 'llama-2-7b-chat-hf' in args.model_name
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            ccp_model = WhiteboxModel(base_model, tokenizer, model_path=args.model_name)
            calc_claim_extractor = ClaimsExtractor(OpenAIChat(
                "openai/gpt-4o",
                base_url="https://openrouter.ai/api/v1",
            ))
            calc_claim_nli = GreedyAlternativesNLICalculator(Deberta(device="cpu"))
            estimator = ClaimConditionedProbabilityClaim()
        elif args.method == 'verify_with_rag':
            from openai import OpenAI
            from flashrag.config import Config
            from flashrag.utils import get_retriever
            
            client = OpenAI(
                base_url=args.base_url,  # 'https://openrouter.ai/api/v1',
                api_key=os.environ.get('OPENROUTER_API_KEY'),
            )
            
            config_dict = {
                'index_path': 'rag_corpus/bm25',
                'corpus_path': 'rag_corpus/corpus_20251105.jsonl',
                'retrieval_method': 'bm25',
                'retrieval_topk': 3,
            }

            config = Config(config_dict=config_dict)
            retriever = get_retriever(config)

            sentences = ds['sentences']

        def get_nli_pred(examples):
            results = {
                'sentences': [],
                'answer': [],
            }
            if args.do_decontextualize:
                results['original_answer'] = examples['answer']

            for i in range(len(examples['question'])):
                question = examples['question'][i]
                answer = examples['answer'][i]
                query = examples['query'][i] if 'query' in examples else None
                document = examples['document'][i] if 'document' in examples else None
                sentences = examples['sentences'][i]

                results['answer'].append(answer)
                # print('debug: decontextualized answer:', answer)
                # print('debug: decontextualized sentences:', sentences)

                if 'range' in sentences[0]:
                    sentence_ranges = []
                    for i, s in enumerate(sentences):
                        if i < len(sentences) - 1:
                            sentence_ranges.append((s['range'][0], sentences[i+1]['range'][0]))
                        else:
                            sentence_ranges.append((s['range'][0], len(answer)))
                else:
                    sentence_ranges = []
                    offset = 0
                    for i, s in enumerate(sentences):
                        start = answer.index(s['text'], offset)
                        end = start + len(s['text'])
                        offset = end
                        sentence_ranges.append((start, end))

                if args.method == 'lynx':
                    nli_results = do_lynx_predict(
                        question,
                        answer, 
                        query, 
                        document,
                        sentence_ranges=sentence_ranges,
                        base_url=args.base_url,
                        nli_model=args.model_name
                    )
                elif args.method == 'minicheck':
                    nli_results = do_minicheck_predict(
                        minicheck_scorer,
                        question,
                        answer,
                        query,
                        document,
                        sentence_ranges=sentence_ranges
                    )
                elif args.method == 'anahv2':
                    nli_results = do_anahv2_predict(
                        question,
                        answer,
                        query,
                        document,
                        sentence_ranges=sentence_ranges,
                        base_url=args.base_url,
                        nli_model=args.model_name
                    )
                elif args.method == 'zero_shot':
                    if args.fine_grained_annotation:
                        nli_results = do_zero_shot_predict(
                            question, 
                            answer, 
                            sentence_ranges,
                            base_url=args.base_url,
                            model_name=args.model_name,
                        )
                    else:
                        raise NotImplementedError
                elif args.method == 'infuse':
                    nli_results = do_infuse_predict(
                        question,
                        answer,
                        sentence_ranges=sentence_ranges,
                        model=infuse_model,
                    )
                elif args.method == 'ccp':
                    nli_results = do_ccp_predict(
                        question,
                        answer,
                        sentence_ranges=sentence_ranges,
                        ccp_model=ccp_model,
                        calc_claim_extractor=calc_claim_extractor,
                        calc_claim_nli=calc_claim_nli,
                        estimator=estimator,
                    )
                elif args.method == 'verify_with_rag':
                    # assert sentences[0].get('nli_pred', None) is not None, 'Please provide the original zero-shot NLI predictions in the input sentences.'
                    nli_results = do_verify_with_rag_predict(
                        question,
                        answer,
                        sentences=sentences,
                        client=client,
                        nli_model=args.model_name,
                        retriever=retriever,
                    )
                else:
                    raise ValueError(f'Method {args.method} not supported')
                
                for j in range(len(sentences)):
                    sentences[j]['nli_pred'] = nli_results['nli_pred'][j]
                    sentences[j]['nli_note'] = nli_results['note'][j] if 'note' in nli_results else ''
                results['sentences'].append(sentences)
            return results

        ds = ds.map(get_nli_pred, batched=True, batch_size=1, remove_columns=['sentences', 'answer'], load_from_cache_file=False)
        ds.to_json(f'{args.method}_results/{args.dataset}_sentence_level_results.jsonl', orient='records', lines=True, force_ascii=False)

    if 'annotation' in ds['sentences'][0][0]:
        if args.fine_grained_annotation:
            fine_grained_evaluate(ds)
        else:
            evaluate(ds, args.uncertainty_thresh)