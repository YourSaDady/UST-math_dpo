# 1.1 给UltraChat数据集加上权重mask
from transformers import AutoTokenizer
import json
import torch
import tqdm

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.pad_token = tokenizer.eos_token
input_root = "/home/nlpintern1/songyc/datasets/UltraFeedbackPair_R1_0521/rl/"
output_root = "/home/nlpintern1/songyc/datasets/UltraFeedbackPair_Masked_0527/rl/"
# /home/nlpintern1/songyc/datasets/UltraFeedbackPair_R1_0521/evaluation/0.json

for file_idx in range(4):
    output_list = []
    with open(input_root + str(file_idx) + ".json", "r", encoding='utf-8') as inf:
        dt = json.load(inf)
        for pair in dt:
            # print("pair: \n", pair)
            rejected_tok = tokenizer(
                pair['rejected'], 
                padding="longest", 
                # truncation=True, 
                # max_length=1024
            )
            chosen_tok = tokenizer(
                pair['chosen'], 
                padding="longest", 
                # truncation=True, 
                # max_length=1024
            )
            new_pair = {
                'prompt': pair['prompt'], 
                'rejected': pair['rejected'], 
                'chosen': pair['chosen'],
                'rejected_mask': [1.0 for _ in range(len(rejected_tok['input_ids']))],
                'chosen_mask': [1.0 for _ in range(len(chosen_tok['input_ids']))]
            }
            # print("rejected_tok: ", rejected_tok)
            # print("chosen_tok: ", chosen_tok)
            # print("new pair: \n", new_pair)
            output_list.append(new_pair)
    with open(output_root + str(file_idx) + ".json", "w") as outf:
        json.dump(output_list, outf, ensure_ascii=False)