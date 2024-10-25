N_GPUS=4
# MODEL_NAME=Llama-7b-hf
# MODEL_NAME=open_llama_3b_v2
MODEL_NAME=pythia-1b-deduped
# MODEL_NAME=TinyLlama-1.1B-intermediate-step-1431k-3T
# MODEL_NAME=R1_sft022_0305_1e_4_EM_0_30
# DATASET=hh-rlhf
DATASET=UltraFeedback
SUFFIX=UNMASKED
REPEATS=1
# export CUDA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES=0
python spin/prepare_dataset.py \
  --data_frac 0 \
  --repeats ${REPEATS} \
  --n_gpus ${N_GPUS} \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET} \
  --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=5
export CUDA_VISIBLE_DEVICES=1
python spin/prepare_dataset.py \
  --data_frac 1 \
  --repeats ${REPEATS} \
  --n_gpus ${N_GPUS} \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET} \
  --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=2
python spin/prepare_dataset.py \
  --data_frac 2 \
  --repeats ${REPEATS} \
  --n_gpus ${N_GPUS} \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET} \
  --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=3
python spin/prepare_dataset.py \
  --data_frac 3 \
  --repeats ${REPEATS} \
  --n_gpus ${N_GPUS} \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET} \
  --data_suffix ${SUFFIX} &





# N_GPUS=4
# MODEL_NAME=TinyLlama-1.1B-intermediate-step-1431k-3T
# DATASET=GSM8K
# SUFFIX=R1
# REPEATS=1
# export CUDA_VISIBLE_DEVICES=0
# python spin/prepare_dataset.py \
#   --data_frac 0 \
#   --repeats ${REPEATS} \
#   --n_gpus ${N_GPUS} \
#   --model_name ${MODEL_NAME} \
#   --dataset_name ${DATASET} \
#   --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=1
# python spin/prepare_dataset.py \
#   --data_frac 1 \
#   --repeats ${REPEATS} \
#   --n_gpus ${N_GPUS} \
#   --model_name ${MODEL_NAME} \
#   --dataset_name ${DATASET} \
#   --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=2
# python spin/prepare_dataset.py \
#   --data_frac 2 \
#   --repeats ${REPEATS} \
#   --n_gpus ${N_GPUS} \
#   --model_name ${MODEL_NAME} \
#   --dataset_name ${DATASET} \
#   --data_suffix ${SUFFIX} &

# export CUDA_VISIBLE_DEVICES=3
# python spin/prepare_dataset.py \
#   --data_frac 3 \
#   --repeats ${REPEATS} \
#   --n_gpus ${N_GPUS} \
#   --model_name ${MODEL_NAME} \
#   --dataset_name ${DATASET} \
#   --data_suffix ${SUFFIX} &