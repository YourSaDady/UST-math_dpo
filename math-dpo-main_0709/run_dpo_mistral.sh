BASE=/home/nlpintern1/liyu
DATASET="${BASE}/dataset/GSM8KPair_sft0310_mistral_0310"
source ~/anaconda3/bin/activate liyu_mistral
# export CUDA_VISIBLE_DEVICES=8,9
# accelerate launch --gpu_ids 8,9 --num_processes 2 --main_process_port 29600 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_llama2.py \
#   --output_dir=./0304_FG_1e_4 \
#   --max_steps=200 \
#   --data_root=${DATASET} \
#   --per_device_train_batch_size=4 \
#   --per_device_eval_batch_size=1 \
#   --learning_rate=1e-4 \
#   --model_name_or_path="${BASE}/models/sft_0222"

export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --gpu_ids 4,5,6,7 --num_processes 4 --main_process_port 29800 --config_file ${BASE}/.cache/accelerate.yaml spin/dpo_mistral.py \
  --output_dir=./mistral \
  --max_steps=1000 \
  --data_root=${DATASET} \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=1 \
  --learning_rate=1e-6 \
  --model_name_or_path="${BASE}/models/sft_0310_mistral"

  # 29895 / (4*8)=934