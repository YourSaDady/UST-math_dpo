from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
import argparse
import time
import os
base_path = "/home/nlpintern1/songyc"
model_path = "/home/nlpintern1/songyc"
eval_size = 4

def evenly_divide_list(lst, num_groups): # 每组q(+1)个list的element
    n = len(lst)
    q, r = divmod(n, num_groups)
    group_sizes = [q] * num_groups
    for i in range(r):
        group_sizes[i] += 1
    
    groups = []
    i = 0
    for size in group_sizes:
        group = lst[i:i+size]
        groups.append(group)
        i += size
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_frac', type=int)
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--model_name', type=str)
    parser.add_argument("--dataset_name", type=str, help="MetaMathQA or GSM8K")
    parser.add_argument("--data_suffix", type=str)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    
    data_frac = args.data_frac
    n_gpus = args.n_gpus
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_suffix = args.data_suffix
    repeats = args.repeats
    date = time.strftime("%m%d", time.localtime())
    
    assert data_frac < n_gpus
    assert eval_size % n_gpus == 0

    # dataset = load_dataset(f'{base_path}/datasets/{dataset_name}/')#############
    # prompts_all = dataset['train']['query'] * repeats
    # response_all = dataset['train']['response'] * repeats
    prompts_all = []
    response_all = []
    input_file = f'{base_path}/datasets/{dataset_name}/dpo.json'
    with open(input_file, 'r') as f:
        data = json.load(f)
        for pair in data:
            prompts_all.append(pair['query'])
            response_all.append(pair['response'])
    print("\nlen(prompts_all) = ",  len(prompts_all), ", len(response_all) = ",  len(response_all))
    assert len(prompts_all) == len(response_all)

    prompts_grouped = evenly_divide_list(prompts_all, n_gpus)
    response_grouped = evenly_divide_list(response_all, n_gpus)
    
    prompts = [f"Question: {question}\nAnswer:" for question in prompts_grouped[data_frac]]
    responses = response_grouped[data_frac]
    
    sampling_params = SamplingParams(max_tokens=512)
    llm = LLM(
        model=f"{model_path}/models/{model_name}/",##########################
    )
    
    start=time.time()
    
    results_gathered = llm.generate(prompts, sampling_params)
    
    timediff=time.time() - start
    print(f"time elapsed: {timediff}") #所有GPU同时各生成一个response所需的时间
    
    results_wrong, results_correct = [], []
    for output, chosen_ans in zip(results_gathered, responses):
        output_ans_text = output.outputs[0].text

        correct_ans = chosen_ans
        # correct_ans = chosen_ans[chosen_ans.find("The answer is: ") + 15:].strip()
        # output_ans_idx = output_ans_text.find("The answer is: ")
        output_ans = output_ans_text
        # output_ans = output_ans_text[output_ans_idx + 15:].strip()
        
        # if output_ans_idx == -1 or output_ans != correct_ans:
        if output_ans != correct_ans:      
            results_wrong.append({
                    "prompt": output.prompt,
                    "rejected": output_ans_text,
                    "chosen": chosen_ans
                })
        else:
            results_correct.append({
                    "prompt": output.prompt,
                    "rejected": output_ans_text,
                    "chosen": chosen_ans
                })


    output_root = f"/home/nlpintern1/songyc/datasets/{dataset_name}Pair_{data_suffix}_{date}/"
    
    if not os.path.exists(output_root):
        os.mkdir(output_root)
        os.mkdir(output_root + "rl/")
        os.mkdir(output_root + "evaluation/")
    if eval_size > 0:
        eval_size_per_frac = int(eval_size / n_gpus)
        eval_set = results_wrong[:eval_size_per_frac]
        train_set = results_wrong[eval_size_per_frac:]
        with open(output_root + f"evaluation/{data_frac}.json", "w", encoding='utf-8') as f:
            json.dump(eval_set, f, ensure_ascii=False)
    else:
        train_set = results_wrong

    with open(output_root + f"rl/{data_frac}.json", "w", encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False)
    with open(output_root + "correct.json", "w") as f:
        json.dump(results_correct, f, ensure_ascii=False)#######为什么正确的dataset不区分data_frac?