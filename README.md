# Hallucination Detection via Attention Signal Analysis

This repository contains code for detecting hallucinations in large language models using attention signal analysis with Fourier transform, Laplacian operator, and Wavelet transform.

## Dataset

The dataset is provided in the `dataset_example` directory:

(Here, we only take Llama-7b for an example dataset.)

- **RAGTruth dataset** (`dataset_example/ragtruth/`):
  - `anno-Summary-7b.jsonl`: Summary task annotations
  - `anno-QA-7b.jsonl`: Question-answering task annotations  
  - `anno-Data2txt-7b.jsonl`: Data-to-text task annotations

- **HalluRAG dataset** (`dataset_example/hallurag/`):
  - `anno_hallurag_7b.jsonl`: Hallucination annotations

Each data entry contains:
- `document`: Input context/document
- `response`: Model-generated response
- `labels`: List of hallucination labels with character positions
- `problematic_spans`: List of problematic text spans
- `split`: Data split (train/test)

**Note**: The example dataset contains only the first 20 entries per file for demonstration purposes.

## Setup

1. Install required dependencies:
```bash
pip install torch transformers numpy scikit-learn tqdm editdistance pytorch-wavelets
```

**Note**: `pytorch-wavelets` is required for the wavelet transform method.


## Usage

### Step 1: Extract Attention Patterns

Extract attention patterns from the model using one of three methods:

#### Fourier Transform
```bash
python step01_extract_attns_fourier.py \
    --data-type summary \
    --data-path dataset_example/ragtruth/anno-Summary-7b.jsonl \
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
    --data-path dataset_example/ragtruth/anno-Summary-7b.jsonl \
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
    --data-path dataset_example/ragtruth/anno-Summary-7b.jsonl \
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
- `--max-new-tokens`: Maximum number of tokens to generate (default: 3000)
- `--top_p`, `--top_k`, `--temperature`: Generation parameters
- `--do-sample`: Enable sampling during generation
- `--max-memory`: Maximum memory per GPU in GiB (default: 45)

### Step 2: Train Classifier

Train a logistic regression classifier using the extracted attention features:

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
- `--anno_2`: Task type for transfer evaluation
- `--attn_2`: Path to attention features for transfer evaluation
- `--tokenizer_name`: Tokenizer name (should match the model used in Step 1)
- `--ifft_mode`: Feature combination mode (default: `new+context`). Combines new token features and context features.
- `--f_cutoff`: Frequency cutoff for Fourier transform (Fourier only, default: 0.45)

**Output:**
The script prints test macro metrics (AUROC, Precision, Recall, F1, Accuracy) and transfer macro metrics (evaluated on the second dataset).

## Notes

- The code uses teacher forcing to extract attention patterns for ground-truth responses, which requires modifying the transformers library as mentioned in the Setup section.
- The example dataset contains only 20 entries per file. For full experiments, replace with complete datasets.
- Model paths and data paths can be modified in the respective script files.
