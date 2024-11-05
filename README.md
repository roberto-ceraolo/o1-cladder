# o1-cladder

Evaluation of newly introduced OpenAI _o1_ models on CLadder. This repository contains the code to evaluate o1 models on CLadder, the benchmark introduced by Z. Jin et al. in [CLadder: Assessing Causal Reasoning in Language Models](https://arxiv.org/abs/2312.04350).

## Data Source

The data is taken from the CLadder repository, available [here](https://github.com/causalNLP/cladder?tab=readme-ov-file). More specifically, the data for [here](https://edmond.mpg.de/dataset.xhtml?persistentId=doi%3A10.17617%2F3.NVRRA9) is the download link. We used the first 1000 samples of the file `cladder-v1-q-balanced_rand.json`.

## Results

The results are the following. See the pdf report for a more detailed analysis of the performance.

| Model | Overall Accuracy |
|--------|-----------------|
| o1-preview | **86.50** |
| o1-mini | 85.40 |
| o1-mini + C-CoT | 77.00 |
| GPT4 + C-CoT | 70.40 |