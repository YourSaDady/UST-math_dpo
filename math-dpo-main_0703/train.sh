# #!/bin/bash
# source conda activate songyc_lmflow
# BASE=/home/nlpintern1/songyc
# # BASE=/home/nlpintern1/liyu
# DATE=$(date '+%m-%d')s

# EXP="${DATE}" ###########????????
# # EXP="${DATE}0314Full1E_4_FG"
# # DATASET="${BASE}/datasets/hh-rlhf"
# DATASET="${BASE}/datasets/UltraFeedbackPair_UNMASKED_0527"
# # DATASET="${BASE}/datasets/GSM8KPair_R1_0520"
# MODEL=pythia-1b-deduped
# # MODEL="0313_sft_llama_full" # pretrain模型
# # MODEL=TinyLlama-1.1B-intermediate-step-1431k-3T
# OUTPUT_DIR=${BASE}/finer_dpo/math-dpo-main/math-dpo-main/results/ultrachat_compare/without_mask
# # OUTPUT_DIR=${BASE}/results/${EXP}


#!/bin/bash
source activate mygo_lmflow
BASE=/u/momoka
DATE=$(date '+%m-%d')s

EXP="${DATE}0314Full1E_4_FG"
DATASET="${BASE}/datasets/UltraFeedbackPair_MASKED_0527"
# DATASET="${BASE}/dataset/GSM8KPair_0314_0314_FG"
MODEL="pythia-1b-deduped" # pretrain模型
OUTPUT_DIR=${BASE}/math-dpo-main_0703/results/ultrachat_compare/with_mask
# OUTPUT_DIR=${BASE}/results/${EXP}

#训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export CUDA_VISIBLE_DEVICES=0
# accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes 10 --main_process_port 29950 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes 1 --main_process_port 29952 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
  --output_dir=${OUTPUT_DIR} \
  --max_steps=234 \
  --data_root=${DATASET} \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=1 \
  --learning_rate=1e-5 \
  --model_name_or_path="${BASE}/models/${MODEL}"
#  ( 1866 * 4) / (4*10)=186.6

# # #合并
# # BASE_MODEL="${BASE}/models/${MODEL}"
# # ADAPTER_NAME="${OUTPUT_DIR}/final_checkpoint"
# # MERGED_NAME="${EXP}_merged"

# # # python spin/merge_peft_adapter.py \
# # #   --base_model_name=${BASE_MODEL} \
# # #   --adapter_model_name=${ADAPTER_NAME} \
# # #   --output_name=${MERGED_NAME}


# # #评测
# # source conda activate liyu_lm_eval_harness
# # # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# # nohup lm_eval --model hf \
# #     --model_args pretrained=${MERGED_NAME},tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8, \
# #     --tasks gsm8k \
# #     --output_path ${BASE}/finer_dpo/math-dpo-main/math-dpo-main/${MERGED_NAME}/eval.json \
# #     --device cuda:0 \
# #     --log_samples \
# #     --batch_size 4 > ${BASE}/logs/${MERGED_NAME}.log &


