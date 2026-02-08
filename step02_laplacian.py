"""
Train classifier using Laplacian operator features for hallucination detection.
Only outputs test macro and transfer macro metrics.
"""

import editdistance as ed
import numpy as np
import json
import torch
import transformers
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


jsonl_path_dict = {
    "summary": "dataset_example/ragtruth/anno-Summary-7b.jsonl",
    "qa": "dataset_example/ragtruth/anno-QA-7b.jsonl",
    "data2txt": "dataset_example/ragtruth/anno-Data2txt-7b.jsonl",
}


def min_edit_distance_substring(s1, s2):
    """Find minimum edit distance substring."""
    len_s1 = len(s1)
    min_edit_dist = float('inf')
    best_substring = None
    assert len(s2) >= len_s1, "s2 must be longer than s1"

    for i in range(len(s2) - len_s1 + 1):
        sub_s2 = s2[i:i + len_s1]
        dist = ed.eval(s1, sub_s2)
        if dist < min_edit_dist:
            min_edit_dist = dist
            best_substring = sub_s2
    return best_substring, min_edit_dist


def load_files(anno_file, attn_file, tokenizer_name=None, auth_token=None, ifft_mode=None):
    """Load annotation and attention files."""
    if "mistral" in tokenizer_name.lower():
        tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_name, token=auth_token)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, token=auth_token)
    
    def manual_offset_mapping(text, token_ids, tokenizer):
        """Manually compute offset mapping for LlamaTokenizer."""
        decoded_tokens = []
        for token_id in token_ids:
            decoded_tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))

        offsets = []
        running_start = 0
        cur_text = text
        for t in decoded_tokens:
            t_stripped = t.lstrip() if cur_text.startswith(" ") else t
            t_len = len(t_stripped)
            idx = cur_text.find(t_stripped)
            if idx == -1:
                idx = cur_text.find(t)
                t_len = len(t)
                if idx == -1:
                    offsets.append((running_start, running_start))
                    continue
            start = running_start + idx
            end = start + t_len
            offsets.append((start, end))
            running_start = end
            cur_text = text[running_start:]
        return offsets

    anno_data = []
    attn_data = []

    anno_path = jsonl_path_dict[anno_file]
    attn_path = attn_file
    
    with open(anno_path, 'r') as f:
            for line in f:
                anno_data.append(json.loads(line))
    attn_data.extend(torch.load(attn_path))

    if "laplacian" in ifft_mode:
        if "l2" in ifft_mode:
            attn_feature_key = 'context_laplacian_l2_score'
            ifft_high_ratio_key = 'context_laplacian_l2_score'
            ifft_low_ratio_key = 'new_tokens_laplacian_l2_score' 
        else:
            attn_feature_key = 'all_previous_laplacian_score'
            ifft_high_ratio_key = 'context_laplacian_score'
            ifft_low_ratio_key = 'new_tokens_laplacian_score' 

    attn_tensor = []
    ifft_low_ratio_tensor = []
    ifft_high_ratio_tensor = []
    labels = []
    splits = []
    skipped_examples = 0
    
    for idx in range(len(anno_data)):
        if len(anno_data[idx]['labels']) > 0:
            is_hallu = True
        else:
            is_hallu = False
        this_split = anno_data[idx]['split']
        
        if is_hallu:
            if "mistral" in tokenizer_name.lower():
                tokenized_hallucination = tokenizer(anno_data[idx]['response'])
                hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
                hallucination_token_offsets = manual_offset_mapping(
                    anno_data[idx]['response'], hallucination_text2ids, tokenizer)
            else:
                tokenized_hallucination = tokenizer(
                    anno_data[idx]['response'], return_offsets_mapping=True)
                hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
                hallucination_token_offsets = tokenized_hallucination['offset_mapping'][1:]
            
            hallucination_attn_ids = attn_data[idx]['model_completion_ids'].tolist()
            if hallucination_attn_ids[-1] == 2:
                hallucination_attn_ids = hallucination_attn_ids[:-1]
            
            mismatch = False
            if not hallucination_text2ids == hallucination_attn_ids:
                best_substring, min_edit_dist = min_edit_distance_substring(
                    hallucination_text2ids, hallucination_attn_ids) if len(
                    hallucination_text2ids) < len(hallucination_attn_ids) else min_edit_distance_substring(
                    hallucination_attn_ids, hallucination_text2ids)
                if min_edit_dist < 5:
                    mismatch = True
                else:
                    skipped_examples += 1
                    continue
            
            hallucinated_spans = anno_data[idx]['problematic_spans']
            hallucinated_spans_token_offsets = []
            for span_text in hallucinated_spans:
                if not span_text in anno_data[idx]['response']:
                    if len(span_text) > len(anno_data[idx]['response']):
                        span_text = anno_data[idx]['response']
                    else:
                        best_substring, min_edit_dist = min_edit_distance_substring(
                            span_text, anno_data[idx]['response'])
                        span_text = best_substring
                
                span_start_char_pos = anno_data[idx]['response'].index(span_text)
                span_end_char_pos = span_start_char_pos + len(span_text)
                
                span_start_token_pos = -1
                span_end_token_pos = -1
                for i, (start_char_pos, end_char_pos) in enumerate(hallucination_token_offsets):
                    if end_char_pos >= span_start_char_pos and span_start_token_pos == -1:
                        span_start_token_pos = i
                    if end_char_pos >= span_end_char_pos and span_end_token_pos == -1:
                        span_end_token_pos = i
                        break

                assert span_start_token_pos != -1 and span_end_token_pos != -1
                hallucinated_spans_token_offsets.append((span_start_token_pos, span_end_token_pos))
            
            if len(hallucinated_spans_token_offsets) == 0:
                skipped_examples += 1
                continue

            tmp_attn_tensor = []
                tmp_ifft_low_ratio_tensor = []
                tmp_ifft_high_ratio_tensor = []
                for i, (s, e) in enumerate(hallucinated_spans_token_offsets):
                    if i == 0 and s > 0:
                    attn_tensor.append(attn_data[idx][attn_feature_key][:, :, :s])
                    ifft_low_ratio_tensor.append(attn_data[idx][ifft_low_ratio_key][:, :, :s])
                    ifft_high_ratio_tensor.append(attn_data[idx][ifft_high_ratio_key][:, :, :s])
                        labels.append(1)
                        splits.append(this_split)
                
                tmp_attn_tensor.append(attn_data[idx][attn_feature_key][:, :, s:e+1])
                tmp_ifft_low_ratio_tensor.append(attn_data[idx][ifft_low_ratio_key][:, :, s:e+1])
                tmp_ifft_high_ratio_tensor.append(attn_data[idx][ifft_high_ratio_key][:, :, s:e+1])
            
            attn_tensor.append(torch.cat(tmp_attn_tensor, dim=-1))
                ifft_low_ratio_tensor.append(torch.cat(tmp_ifft_low_ratio_tensor, dim=-1))
                ifft_high_ratio_tensor.append(torch.cat(tmp_ifft_high_ratio_tensor, dim=-1))
                labels.append(0)
                splits.append(this_split)
            
                if e < len(hallucination_token_offsets) - 1:
                attn_tensor.append(attn_data[idx][attn_feature_key][:, :, e+1:])
                ifft_low_ratio_tensor.append(attn_data[idx][ifft_low_ratio_key][:, :, e+1:])
                ifft_high_ratio_tensor.append(attn_data[idx][ifft_high_ratio_key][:, :, e+1:])
                    labels.append(1)
                    splits.append(this_split)
            else:
            attn_tensor.append(attn_data[idx][attn_feature_key])
                ifft_low_ratio_tensor.append(attn_data[idx][ifft_low_ratio_key])
                ifft_high_ratio_tensor.append(attn_data[idx][ifft_high_ratio_key])
                    labels.append(1)
                    splits.append(this_split)
    
        labels = np.array(labels)
    return attn_tensor, ifft_low_ratio_tensor, ifft_high_ratio_tensor, labels, splits


def extract_time_series_features(attn_tensor):
    """Extract features from time series."""
    features = []
    num_examples = len(attn_tensor)
    
    for i in tqdm(range(num_examples)):
        example = attn_tensor[i].clone()
        try:
            example = example.view(-1, example.shape[2])
        except:
            continue
        example = example.transpose(0, 1)
        feature_vector = example.mean(dim=0).numpy()
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feature_vector)

    return np.array(features)


def calculate_prf_acc_with_threshold(labels, pred_proba, threshold, mode='macro'):
    """Calculate precision, recall, f1, accuracy using a specific threshold."""
    preds = (pred_proba >= threshold).astype(int)
    precision = precision_score(labels, preds, pos_label=0, average=mode)
    recall = recall_score(labels, preds, pos_label=0, average=mode)
    f1 = f1_score(labels, preds, pos_label=0, average=mode)
    accuracy = accuracy_score(labels, preds)
    return precision, recall, f1, accuracy


def find_best_threshold_on_validation(y_val, y_val_proba, mode='macro', search_step=0.1):
    """Find best threshold using validation set."""
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0, 1.01, search_step):
        preds_threshold = (y_val_proba >= threshold).astype(int)
        f1 = f1_score(y_val, preds_threshold, pos_label=0, average=mode)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def main(anno_file_1, attn_file_1, anno_file_2, attn_file_2, 
         tokenizer_name=None, output_path=None, auth_token=None, ifft_mode=None, f_cutoff=None):
    """Main function for training and evaluation."""
    print(f"======== Loading data from {anno_file_1} and {attn_file_1}...")
    attn_tensor, ifft_low_ratio, ifft_high_ratio, labels, splits = load_files(
        anno_file_1, attn_file_1, tokenizer_name=tokenizer_name, auth_token=auth_token, ifft_mode=ifft_mode)
    
    print(f"Loaded: {len(attn_tensor)} examples")
    
    new_features = extract_time_series_features(ifft_low_ratio)
    context_features = extract_time_series_features(ifft_high_ratio)
    time_series_features = np.concatenate([new_features, context_features], axis=1)
    
            X_train, X_test, y_train, y_test = [], [], [], []
            for i in range(len(time_series_features)):
                if splits[i] == 'train' or "train" in splits[i]:
                    X_train.append(time_series_features[i])
                    y_train.append(labels[i])
                elif splits[i] == 'test' or "test" in splits[i]:
                    X_test.append(time_series_features[i])
                    y_test.append(labels[i])
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, 
        stratify=y_train if len(set(y_train)) > 1 else None
    )
                    
                classifier = LogisticRegression(max_iter=1000)
                classifier.fit(X_tr, y_tr)

                y_val_proba = classifier.predict_proba(X_val)[:, 1]
                best_threshold = find_best_threshold_on_validation(y_val, y_val_proba, mode='macro')
    
                y_test_proba = classifier.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_test_proba)
    precision, recall, f1, accuracy = calculate_prf_acc_with_threshold(
        y_test, y_test_proba, best_threshold, mode='macro')
    
    print("\n======== Test Results ========")
    print(f"AUROC: {auroc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Transfer evaluation
    print(f"\n======== Transfer to {anno_file_2} and {attn_file_2}...")
    transfer_attn_tensor, transfer_ifft_low_ratio, transfer_ifft_high_ratio, transfer_labels, transfer_splits = load_files(
        anno_file_2, attn_file_2, tokenizer_name=tokenizer_name, auth_token=auth_token, ifft_mode=ifft_mode)
    
    transfer_new_features = extract_time_series_features(transfer_ifft_low_ratio)
    transfer_context_features = extract_time_series_features(transfer_ifft_high_ratio)
    transfer_time_series_features = np.concatenate([transfer_new_features, transfer_context_features], axis=1)
    
    transfer_X_test, transfer_y_test = [], []
    for i in range(len(transfer_time_series_features)):
        if 'test' in transfer_splits[i]:
            transfer_X_test.append(transfer_time_series_features[i])
            transfer_y_test.append(transfer_labels[i])
    
    transfer_X_test = np.array(transfer_X_test)
    transfer_y_test = np.array(transfer_y_test)
    
    y_pred_proba_transfer = classifier.predict_proba(transfer_X_test)[:, 1]
    transfer_auroc = roc_auc_score(transfer_y_test, y_pred_proba_transfer)
    transfer_precision, transfer_recall, transfer_f1, transfer_accuracy = calculate_prf_acc_with_threshold(
        transfer_y_test, y_pred_proba_transfer, best_threshold, mode='macro')
    
    print("\n======== Transfer Results ========")
    print(f"AUROC: {transfer_auroc:.4f}")
    print(f"Precision: {transfer_precision:.4f}, Recall: {transfer_recall:.4f}, F1: {transfer_f1:.4f}, Accuracy: {transfer_accuracy:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train classifier using Laplacian features.")
    
    parser.add_argument('--anno_1', type=str, default='summary')
    parser.add_argument('--attn_1', type=str, default='outputs/attn-features-summary-7b.pt')
    parser.add_argument('--anno_2', type=str, default='qa')
    parser.add_argument('--attn_2', type=str, default='outputs/attn-features-qa-7b.pt')
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--auth_token', type=str, default=None)
    parser.add_argument('--ifft_mode', type=str, default='new+context')
    parser.add_argument('--f_cutoff', type=float, default=None)

    args = parser.parse_args()
    
    main(
        args.anno_1, args.attn_1, args.anno_2, args.attn_2,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path,
        auth_token=args.auth_token,
        ifft_mode=args.ifft_mode,
        f_cutoff=args.f_cutoff,
    )
