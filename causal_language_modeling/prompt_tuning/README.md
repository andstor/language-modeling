# Prompt Tuning

## Setup

```shell
pip install -r requirements.txt 
```

## Usage

```bash
python script/run_clm_prompt_tuning.py \
    --model_name_or_path gpt2 \
    --dataset_name andstor/vulnerable_smart_contracts \
    --dataset_config_name manual_all_s5_embedded \
    --adapter_name default \
    --num_virtual_tokens 4 \
    --virtual_tokens_init_text "this is a test" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --output_dir .data/tmp/test-clm
```