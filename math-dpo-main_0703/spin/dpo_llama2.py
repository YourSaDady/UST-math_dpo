# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from utils.dataset import get_MMQP, get_stack_exchange_paired, get_sft, load_dataset_w_mask
from utils.args import ScriptArguments
from dpo_pythia_w_mask import DPO_Otto_Trainer#, DPOFinerTrainer
from trl import DPOTrainer

use_flash_attention = False # True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

base = "/home/nlpintern1/liyu/"
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # 0. use flash attention
    if torch.cuda.get_device_capability()[0] >= 8 and use_flash_attention:
        from utils.llama_patch import replace_attn_with_flash_attn
        replace_attn_with_flash_attn()
        print("Using flash attention")

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
    )
    model_ref.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 1.2 check if it's using flash attention
    if use_flash_attention:
        from utils.llama_patch import forward
        assert model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"
        assert model_ref.model.layers[0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"


    # 2. Load the Stack-exchange paired dataset
    # train_dataset = get_stack_exchange_paired(data_dir=script_args.data_dir, 
    #                                           data_root=script_args.data_root,
    #                                           sanity_check=script_args.sanity_check)
    # train_dataset = get_MMQP(data_dir="rl", 
    #                         data_root=script_args.data_root,
    #                         sanity_check=script_args.sanity_check) 

    print("\n\ndata_root: ", script_args.data_root, "\n\n")

    train_dataset = load_dataset_w_mask(data_dir="rl", ######################OSError: [Errno 12] Cannot allocate memory??
                            data_root=script_args.data_root,
                            sanity_check=script_args.sanity_check) 


    # sft_dataset = get_sft(data_root=script_args.data_root)

    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # print("\n\nlength of train dataset after filter: ", len(train_dataset), "\n\n")

    # 3. Load evaluation dataset
    # eval_dataset = get_MMQP(data_dir="evaluation", data_root=script_args.data_root, sanity_check=True)
    eval_dataset = load_dataset_w_mask(data_dir="evaluation", data_root=script_args.data_root, sanity_check=True)
    # print("\n\neval_dataset first element: \n", eval_dataset[0], "\n\n")
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2"
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPO_Otto_Trainer(
    # dpo_trainer = DPOFinerTrainer(
    # dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )
    # 6. train
    # print("\n\nnow print the first batch of train_data_loader: \n\n")
    # dpo_trainer.precompute_ref_log_probs = True
    # train_data_loader = dpo_trainer.get_train_dataloader()

    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)