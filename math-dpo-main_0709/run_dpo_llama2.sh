# conda activate liyu_lmflow
BASE=/home/nlpintern1/songyc
# BASE=/home/nlpintern1/liyu
DATASET="${BASE}/datasets/hh-rlhf"
# DATASET="${BASE}/datasets/GSM8KPair_R1_0518"
# DATASET="${BASE}/dataset/GSM8KPair_R1_0312"
MODEL="TinyLlama-1.1B-intermediate-step-1431k-3T"
# MODEL="R1_sft022_0305_1e_4_EM_0_30"

# export CUDA_VISIBLE_DEVICES=8,9
# accelerate launch --gpu_ids 8,9 --num_processes 2 --main_process_port 29600 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
#   --output_dir=./0304_FG_1e_4 \
#   --max_steps=200 \
#   --data_root=${DATASET} \
#   --per_device_train_batch_size=4 \
#   --per_device_eval_batch_size=1 \
#   --learning_rate=1e-4 \
#   --model_name_or_path="${BASE}/models/sft_0222"

# export CUDA_VISIBLE_DEVICES=6,7
export CUDA_VISIBLE_DEVICES=1,2
accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes 2 --main_process_port 29952 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
# accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes 2 --main_process_port 29950 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
  --output_dir=../finer_dpo/math-dpo-main/math-dpo-main/R2 \
  --max_steps=200 \
  --data_root=${DATASET} \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=1 \
  --learning_rate=1e-4 \
  --model_name_or_path="${BASE}/models/${MODEL}"



  # 29895 / (4*8)=934




# BASE=/home/nlpintern1/liyu
# DATASET="${BASE}/dataset/GSM8KPair_R1_0312"
# MODEL="TinyLlama-1.1B-intermediate-step-1431k-3T"
# # export CUDA_VISIBLE_DEVICES=8,9
# # accelerate launch --gpu_ids 8,9 --num_processes 2 --main_process_port 29600 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
# #   --output_dir=./0304_FG_1e_4 \
# #   --max_steps=200 \
# #   --data_root=${DATASET} \
# #   --per_device_train_batch_size=4 \
# #   --per_device_eval_batch_size=1 \
# #   --learning_rate=1e-4 \
# #   --model_name_or_path="${BASE}/models/sft_0222"

# export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes 2 --main_process_port 29950 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
#   --output_dir=./R2 \
#   --max_steps=200 \
#   --data_root=${DATASET} \
#   --per_device_train_batch_size=4 \
#   --per_device_eval_batch_size=1 \
#   --learning_rate=1e-4 \
#   --model_name_or_path="${BASE}/models/${MODEL}"