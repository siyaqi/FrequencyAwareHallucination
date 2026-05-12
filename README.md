# Hallucination Detection via Attention Signal Analysis

This repository contains code for detecting hallucinations in large language models using attention signal analysis with Fourier transform, Laplacian operator, and Wavelet transform.

## Dataset

Datasets are provided under the `dataset/` directory, organized by source and model:

- **RAGTruth dataset** (`dataset/ragtruth/<model>/`):
  - `anno-Summary-7b.jsonl` / `anno-Summary-13b.jsonl`: Summary task annotations
  - `anno-QA-7b.jsonl` / `anno-QA-13b.jsonl`: Question-answering task annotations
  - `anno-Data2txt-7b.jsonl` / `anno-Data2txt-13b.jsonl`: Data-to-text task annotations

- **HalluRAG dataset** (`dataset/hallurag/<model>/`):
  - `anno_hallurag_7b.jsonl` / `anno_hallurag_13b.jsonl`: Hallucination annotations

Models covered: `llama-2-7b-chat`, `llama-2-13b-chat`, `mistral-7b-instruct`.

Each data entry contains:
- `document`: Input context/document
- `response`: Model-generated response
- `labels`: List of hallucination labels with character positions
- `problematic_spans`: List of problematic text spans
- `split`: Data split (train/test)

## Setup

1. Install required dependencies:
```bash
pip install torch transformers numpy scikit-learn tqdm editdistance pytorch-wavelets
```

**Notes:**
- `pytorch-wavelets` is required for the wavelet transform method.
- The code uses a patched local copy of `transformers==4.32.0` that supports teacher-forcing during generation and a custom `LLamaQaStoppingCriteria`. Add it to `PYTHONPATH` before running:
  ```bash
  export PYTHONPATH="$(pwd)/transformers-4.32.0/src:$PYTHONPATH"
  ```


## Usage

The pipeline has two stages. By default, paths in the scripts use the relative `dataset/ragtruth/llama-2-7b-chat/` data and write/read attention features under `outputs/`. Adjust paths if you want a different model directory.

### Step 1: Extract Attention Patterns

Extract attention patterns from the model using one of three methods:

#### Fourier Transform
```bash
python step01_extract_attns_fourier.py \
    --data-type summary \
    --data-path dataset/ragtruth/llama-2-7b-chat/anno-Summary-7b.jsonl \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --device cuda \
    --num-gpus 1 \
    --output-path outputs/attn-features-summary-7b.pt \
    --auth-token YOUR_HF_TOKEN \
    --f_cutoff 0.45
```

#### Laplacian Operator
```bash
python step01_extract_attns_laplacian.py \
    --data-type summary \
    --data-path dataset/ragtruth/llama-2-7b-chat/anno-Summary-7b.jsonl \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --device cuda \
    --num-gpus 1 \
    --output-path outputs/attn-features-summary-7b.pt \
    --auth-token YOUR_HF_TOKEN
```

#### Wavelet Transform
```bash
python step01_extract_attns_wavelet.py \
    --data-type summary \
    --data-path dataset/ragtruth/llama-2-7b-chat/anno-Summary-7b.jsonl \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --device cuda \
    --num-gpus 1 \
    --output-path outputs/attn-features-summary-7b.pt \
    --auth-token YOUR_HF_TOKEN
```

**Key Parameters:**
- `--data-type`: Task type (`summary`, `qa`, or `data2txt`)
- `--data-path`: Path to JSONL annotation file
- `--model-name`: HuggingFace model name
- `--device`: Device to use (`cuda` or `cpu`)
- `--num-gpus`: Number of GPUs (or `auto`)
- `--output-path`: Output path for saved attention features
- `--auth-token`: HuggingFace authentication token (if needed)
- `--f_cutoff`: Frequency cutoff for Fourier transform (Fourier only, default: 0.45)

**Additional optional parameters:**
- `--debug`: Run on the first 10 entries for a quick smoke test
- `--subsample N`: Keep every N-th entry
- `--start-sample-idx` / `--end-sample-idx`: Process a slice of the dataset
- `--max-new-tokens`: Maximum number of tokens to generate (default: 3000)
- `--top_p`, `--top_k`, `--temperature`: Generation parameters
- `--do-sample`: Enable sampling during generation
- `--max-memory`: Maximum memory per GPU in GiB (default: 45)

### Step 2: Train Classifier

Train a logistic regression classifier using the extracted attention features. Pass `--anno_2` / `--attn_2` to evaluate cross-task transfer; omit them to skip the transfer block and report only the in-task test metrics.

The `jsonl_path_dict` at the top of each `step02_*.py` resolves the `--anno_*` keys (`summary` / `qa` / `data2txt`) to the corresponding JSONL files. Edit it if your data lives under a different model directory.

#### Fourier Transform
```bash
python step02_fourier.py \
    --anno_1 summary \
    --attn_1 outputs/attn-features-summary-7b.pt \
    --anno_2 qa \
    --attn_2 outputs/attn-features-qa-7b.pt \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --ifft_mode new+context \
    --f_cutoff 0.45 \
    --auth_token YOUR_HF_TOKEN
```

#### Laplacian Operator
```bash
python step02_laplacian.py \
    --anno_1 summary \
    --attn_1 outputs/attn-features-summary-7b.pt \
    --anno_2 qa \
    --attn_2 outputs/attn-features-qa-7b.pt \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --ifft_mode new+context \
    --auth_token YOUR_HF_TOKEN
```

#### Wavelet Transform
```bash
python step02_wavelet.py \
    --anno_1 summary \
    --attn_1 outputs/attn-features-summary-7b.pt \
    --anno_2 qa \
    --attn_2 outputs/attn-features-qa-7b.pt \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --ifft_mode new+context \
    --auth_token YOUR_HF_TOKEN
```

**Key Parameters:**
- `--anno_1`: Task type for training data (`summary`, `qa`, or `data2txt`)
- `--attn_1`: Path to attention features for training
- `--anno_2`: Task type for transfer evaluation (omit to skip transfer)
- `--attn_2`: Path to attention features for transfer evaluation (omit to skip transfer)
- `--tokenizer_name`: Tokenizer name (should match the model used in Step 1)
- `--ifft_mode`: Feature combination mode (default: `new+context`). Combines new token features and context features.
- `--f_cutoff`: Frequency cutoff for Fourier transform (Fourier only, default: 0.45)

**Output:**
The script prints test macro metrics (AUROC, Precision, Recall, F1, Accuracy) and — when `--anno_2` / `--attn_2` are provided — transfer macro metrics (evaluated on the second dataset).

### Step 2 (alt): Evaluate a Pre-trained Classifier

Each `step02_*.py` also supports loading a previously trained classifier (`.pkl`) and evaluating it directly on a chosen split — skipping training. Pass `--classifier` to switch into evaluation-only mode.

#### Fourier Transform
```bash
python step02_fourier.py \
    --anno_1 qa \
    --attn_1 outputs/attn-features-qa-7b.pt \
    --classifier classifiers/ragtruth/fourier/llama7b/classifier_7b_qa_sliding_window_1_0.45.pkl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --auth_token YOUR_HF_TOKEN
```

#### Laplacian Operator
```bash
python step02_laplacian.py \
    --anno_1 qa \
    --attn_1 outputs/attn-features-qa-7b.pt \
    --classifier classifiers/ragtruth/laplacian/llama7b/classifier_7b_qa_sliding_window_1_laplacian_l2_context_new_None.pkl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --ifft_mode new+context \
    --auth_token YOUR_HF_TOKEN
```

#### Wavelet Transform
```bash
python step02_wavelet.py \
    --anno_1 qa \
    --attn_1 outputs/attn-features-qa-7b.pt \
    --classifier classifiers/ragtruth/wavelet/llama7b/classifier_7b_qa_sliding_window_1_context_new_None.pkl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --auth_token YOUR_HF_TOKEN
```

**Eval-only parameters:**
- `--classifier`: Path to a trained classifier `.pkl`. When set, the script skips training and evaluates the loaded classifier on the requested split of `anno_1` / `attn_1`. The pickle may be either a bare sklearn classifier or a `{'clf': ...}` dict.
- `--eval_split`: Substring match against the anno `split` field. Default: `test`.
- `--threshold`: Decision threshold for class 1. Omit to pick the F1-best threshold on the eval split.
- `--sliding_window` and (for Laplacian) `--ifft_mode` must match what was used at training time, otherwise the feature dimensionality check will fail.

## Notes

- The code uses teacher forcing to extract attention patterns for ground-truth responses, which relies on the patched `transformers-4.32.0/` shipped with this repo. Make sure it is on `PYTHONPATH`.
- Model paths and data paths can be modified at the top of each script (`jsonl_path_dict`) or via CLI flags.
- All generated artifacts go to `outputs/` and are ignored by git (see `.gitignore`).


