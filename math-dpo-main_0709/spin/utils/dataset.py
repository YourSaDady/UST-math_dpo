from typing import Dict, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1b-deduped"
)
tokenizer.pad_token = tokenizer.eos_token

def load_dataset_w_mask(
    data_dir: str = "rl",
    data_root: str = "/home/nlpintern1/songyc/datasets/UltraFeedbackPair_MASKED_0527", # "dataset/MetaMathQAPair",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """
    Load the dataset wit mask
    """
    dataset = load_dataset( #############?
        data_root,
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    print("\n\ndata_root: ", data_root, "\n\n")
    print("dataset[0]['chosen_mask]: ", dataset[0]['chosen_mask'], "\n\n")

    def return_prompt_and_responses(samples) -> Dict[str, str]: 
        return {
            "prompt": samples['prompt'],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
            "chosen_mask": samples["chosen_mask"],
            "rejected_mask": samples['rejected_mask']
        }

    return dataset.map( #############?
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_MMQP(
    data_dir: str = "rl",
    data_root: str = "/home/nlpintern1/songyc/datasets/UltraFeedbackPair_R1_0521", # "dataset/MetaMathQAPair",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Load the MetaMathPairQA from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        data_root,
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples['prompt'],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


def get_Masked_Dataset(
    data_dir: str = "rl",
    data_root: str = "dataset/UltraFeedbackPair_R1_0521",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Prepare the UltraChat dataset with the corresponding masks.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        data_root,
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples['prompt'],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }
    #def #要不要替换成tokenized？取决于DPOTrainer是否一定会在train的过程中tokenize
    # 应该是可以的
   
    def tokenize_function(samples):
        # print("Here is samples['prompt]:\n", samples["prompt"])
        return {
            "prompt": tokenizer(samples['prompt'], padding="max_length", truncation=True, max_length=1024),
            "chosen": tokenizer(samples['chosen'], padding="max_length", truncation=True, max_length=1024),
            "rejected": tokenizer(samples['rejected'], padding="max_length", truncation=True, max_length=1024)
        } #, truncation=True, max_length=max_seq_length)

    # print('Tokenizing data. This may take a while...')

    return dataset.map(
        # tokenize_function,
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        # input_columns=["prompt", "chosen", "rejected"] 
    )

def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    data_root: str = "../dataset/stack-exchange-paired",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        data_root,
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def get_sft(
        data_root: str = "dataset/MetaMathQAPair",
        num_proc=24,
    ) -> Dataset:
    path = os.path.join(data_root, "correct.json")
    dataset = load_dataset(path)
    original_columns = dataset.column_names
    def return_query_and_answers(samples) -> Dict[str, str]:
        return {
            "query": samples['prompt'],
            "answer": samples["answer"],
        }
    return dataset.map(
            return_query_and_answers,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns
        )