# math-dpo
## 生成样本
```bash
conda activate liyu_vllm
sh run_infer.sh
```
参数:

* N_GPUS
* MODEL_NAME: 模型放在 `/home/nlpintern1/liyu/models/`路径下,只用填模型名就好
* DATASET&SUFFIX: 生成出来的数据集放在 `/home/nlpintern1/liyu/dataset/${DATASET}Pair_${MODEL}_${SUFFIX}`
* REPEATS: 同一个prompt重复生成多少次
* 若修改了GPU的数量,则需要对应修改`data_frac`

## 训练模型
```bash
conda activate liyu_lmflow
# 如果训练mistral (对应transformers版本不一致)
conda activate liyu_mistral
```
参数:
* DATASET
* MODEL: 同上

## 合并和模型评测
1. 合并
```bash
conda activate liyu_lmflow
BASE_MODEL="/home/nlpintern1/liyu/models/sft_0310"
ADAPTER_NAME="./200_sft_0310FG_1e_4/final_checkpoint"
OUTPUT_NAME="200_sft_0310FG_1e_4_merged"

python spin/merge_peft_adapter.py \
  --base_model_name=${BASE_MODEL} \
  --adapter_model_name=${ADAPTER_NAME} \
  --output_name=${OUTPUT_NAME}
```
2. 评测
```bash
conda activate liyu_lm_eval_harness
export CUDA_VISIBLE_DEVICES=8
nohup lm_eval --model hf \
    --model_args pretrained=${OUTPUT_NAME} \
    --tasks gsm8k \
    --output_path /home/nlpintern1/liyu/evaluation/${OUTPUT_NAME}/eval.json \
    --device cuda:0 \
    --log_samples \
    --batch_size 4 > /home/nlpintern1/liyu/logs/${OUTPUT_NAME}.log &
```

或者也可以直接运行`./train.sh`