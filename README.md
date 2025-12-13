# VeriGray

VeriGray is an unfaithfulness detection dataset that has addressed the annotation ambiguity induced by the external world knowledge and linguistic ambiguity. See [our paper](https://arxiv.org/pdf/2510.21118) for more details. This repository provides the scripts to reproduce the experimental results in the paper.


## Requirements

Since our repository assemblies a bunch of baselines, which require diverse environments, we do not make out a unified environment requirements. Instead, we recommend using different environments for different baselines.

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

```
OPENROUTER_API_KEY=<your openrouter api key> python test.py \
    --method <method> \  # choices: zero_shot, lynx, anahv2, minicheck, infuse, ccp, redeep, verify_with_rag, llm_check
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