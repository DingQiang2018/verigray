# VeriGray

VeriGray is an unfaithfulness detection dataset that addresses the annotation ambiguity induced by external world knowledge and linguistic ambiguity. See [our paper](https://arxiv.org/pdf/2510.21118) for more details. This repository provides the scripts to reproduce the experimental results in the paper.


## Requirements

Since our repository assembles a variety of baselines, which require diverse environments, we do not make out a unified environment requirements. Instead, we recommend using different environments for different baselines.

Basic requirements (for zero_shot, lynx, and anahv2):
```
openai==1.102.0
transformers==4.57.0
datasets==2.19.2
```

minicheck: in addition to the basic requirements, install the following package
```
minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main
```

infuse: in addition to the basic requirements, install the dependencies according to its GitHub repo.

ccp:
```
datasets==3.6.0
transformers==4.51.3
numpy==1.26.4
torch==2.8.0
lm-polygraph
```

redeep: install the dependencies according to its GitHub repo

verify_with_rag (corresponding to GPT-5 + RAG in the paper):
```
datasets==4.4.1
openai==2.7.1
flashrag-dev==0.3.0.dev20250925
transformers==4.57.1
```

## Usage

For zero_shot, verify_with_rag, minicheck, infuse, and ccp, run the following command directly.
```
OPENROUTER_API_KEY=<your openrouter api key> python test.py \
    --method <method> \  # choices: zero_shot, verify_with_rag, minicheck, infuse, ccp, lynx, anahv2, redeep, llm_check
    --overwrite \  # store_true: whether overwrite the cached results
    --data-file <data_file> \  # the file of data to evaluate
    --cache-file <cache_file> \ # the file of the cached results
    --dataset <dataset> \  # choices: verigray 
    --n-samples <n_samples>  # the number of samples to evaluate (using a subset of the dataset to evaluate)
    --base-url https://openrouter.ai/api/v1 \  # the base url of LLM service, set to the Openrouter URL by default
    --model-name <model-name>  # e.g., openai/gpt-5
    --uncertainty-thresh <a float>  # only used for ccp, llm-check, and redeep
    --fine-grained-annotation  # store_true: set to True for zero_shot and verify_with_rag only
```

For lynx and anahv2, 

1. Deploy a local openai service of lynx/anahv2 using vllm;
2. Run `test.py` (the command above).

For redeep and llm_check, 

1. Use `data_to_redeep.py` and `data_to_fava_annotated.py` to convert the data into the format that can be processed by the official implementation of ReDEeP and LLM-Check;
2. Use `data_from_redeep.py` and `data_from_fava_annotated.py` to convert the results back to our format;
3. Run the `test.py` with `--cache-file` assigned to the results file.
