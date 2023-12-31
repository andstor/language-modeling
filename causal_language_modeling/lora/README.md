# Prompt Tuning

## Setup

```shell
pip install -r requirements.txt 
```

## Usage

```bash
python run_clm_lora.py \
    --model_name_or_path gpt2 \
    --dataset_name facebook/opt-125m \
    --dataset_config_name manual_all_s5_embedded \
    --adapter_name default \
    --num_virtual_tokens 4 \
    --virtual_tokens_init_text "this is a test" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --block_size 2048 \
    --do_train \
    --do_eval \
    --output_dir .data/tmp/test-clm
```

```bash
accelerate launch run_clm_seq2seq_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name andstor/methods2test_small \
    --dataset_config_name fm+t \
    --text_column_name text \
    --preprocessing_num_workers 10 \
    
    --report_to all \
    --logging_first_step --logging_steps 1 \
    --load_best_model_at_end \
    --output_dir .data/tmp/test-clm \
    --logging_dir ./logs \

    --block_size 2048 \
    
    --do_train \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size 1 \
    
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 256 \
    --per_device_eval_batch_size 1 \
    
    --do_predict \
    --max_predict_samples 10 \
    --predict_with_generate \
    --generation_max_length 256 \
    
    --adapter_name default \
    --rank 32 \
    --lora_alpha 16 \
    --target_modules "q_proj, v_proj" \
    --lora_dropout 0.1 \
    --bias none \
    --learning_rate 5e-3
```


# run_clm_seq2seq_lora.py
```bash
accelerate launch run_clm_seq2seq_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name andstor/methods2test_small \
    --dataset_config_name fm+t \
    --source_column_name source \
    --target_column_name target \
    --preprocessing_num_workers 10 \
    --report_to all \
    --logging_first_step --logging_steps 1 \
    --load_best_model_at_end \
    --output_dir .data/tmp/test-clm \
    --logging_dir ./logs \
    --block_size 2048 \
    --do_train \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size 1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 10 \
    --per_device_eval_batch_size 1 \
    --do_predict \
    --max_predict_samples 10 \
    --adapter_name default \
    --rank 32 \
    --lora_alpha 16 \
    --target_modules "q_proj, v_proj" \
    --lora_dropout 0.1 \
    --bias none \
    --learning_rate 5e-3
```
 # CLM


 ```bash
accelerate launch run_clm_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name andstor/methods2test_small \
    --dataset_config_name fm+t \
    --source_column_name source \
    --target_column_name target \
    --preprocessing_num_workers 10 \
    \
    --report_to all \
    --logging_first_step --logging_steps 1 \
    --load_best_model_at_end \
    --output_dir .data/tmp/test-clm \
    --logging_dir ./logs \
    --log_preditions \
    --log_predition_samples 10 \
    \
    --block_size 2048 \
    \
    --do_train \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size 1 \
    \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 256 \
    --per_device_eval_batch_size 1 \
    \
    --adapter_name default \
    --rank 32 \
    --lora_alpha 16 \
    --target_modules "q_proj, v_proj" \
    --lora_dropout 0.1 \
    --bias none \
    --learning_rate 5e-3
```

# run_clm_lora.py
```bash
accelerate launch run_clm_lora.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name andstor/methods2test_small \
    --dataset_config_name fm+t \
    --text_column_names "source, target" \
    --preprocessing_num_workers 10 \
    \
    --report_to all \
    --logging_first_step --logging_steps 1 \
    --load_best_model_at_end \
    --output_dir .data/tmp/test-clm \
    --logging_dir ./logs \
    --log_preditions \
    --log_predition_samples 10 \
    \
    --block_size 2048 \
    \
    --do_train \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --warmup_ratio=0.1 \
    --per_device_train_batch_size 1 \
    \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 256 \
    --per_device_eval_batch_size 1 \
    \
    --adapter_name default \
    --rank 32 \
    --lora_alpha 16 \
    --target_modules "q_proj, v_proj" \
    --lora_dropout 0.1 \
    --bias none \
    --learning_rate 5e-3
```

## Distributed training

Due to the large size of the model, it is not feasible to train it on a single GPU. The following code shows how to train the model on multiple GPUs using Microsoft's DeepSpeed library.

```
deepspeed --hostfile=hostfile run_clm.py \
--deepspeed ds_zero2_bf16.json \