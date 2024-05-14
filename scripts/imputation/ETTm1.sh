export CUDA_VISIBLE_DEVICES=6

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/imputation" ]; then
    mkdir ./logs/imputation
fi

if [ ! -d "./checkpoints/imputation" ]; then
    mkdir ./checkpoints/imputation
fi

if [ ! -d "./logs/imputation/ETTm1" ]; then
    mkdir ./logs/imputation/ETTm1
fi

if [ ! -d "./checkpoints/imputation/ETTm1" ]; then
    mkdir ./checkpoints/imputation/ETTm1
fi



model_name=LLM4TS_imputation

root_path_name=./dataset/
data_path_name=ETTm1.csv
data_name=ETTm1
model_id=$data_name'_'$model_name

for mask_rate in 0.375
do
for seq_len in 96
do
for label_len in 0
do
for pred_len in 0
do
for d_model in 768
do
for n_heads in 4
do
for e_layers in 4
do
for llm_layers in 3
do
for d_ff in 768
do
for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.001 
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u run_LLM4TS.py \
    --task_name imputation \
    --mask_rate $mask_rate \
    --lradj type1 \
    --random_seed 2021 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --llm ./hf_models/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 1\
    --stride 1\
    --des $exp_des \
    --train_epochs 100\
    --patience 2\
    --target 'CHF' \
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/imputation/ETTm1 \
    > logs/imputation/ETTm1/$mask_rate'_'$model_id'_sl'$seq_len'_pl'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_percent'$percent'_'$exp_des.log 2>&1 
done
done
done
done
done
done
done
done
done
done
done
done
done
done

for mask_rate in 0.125 0.25
do
for seq_len in 96
do
for label_len in 0
do
for pred_len in 0
do
for d_model in 768
do
for n_heads in 4
do
for e_layers in 4
do
for llm_layers in 3
do
for d_ff in 768
do
for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.001 
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u run_LLM4TS.py \
    --task_name imputation \
    --mask_rate $mask_rate \
    --lradj type1 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --text_token_len 0 \
    --llm ./hf_models/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 1\
    --stride 1\
    --des $exp_des \
    --train_epochs 100\
    --patience 2\
    --target 'CHF' \
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --robust 0 \
    --blank 1 \
    --d_prompt 0 \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/imputation/ETTm1 \
    > logs/imputation/ETTm1/$mask_rate'_'$model_id'_sl'$seq_len'_pl'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_percent'$percent'_'$exp_des.log 2>&1 
done
done
done
done
done
done
done
done
done
done
done
done
done
done

for mask_rate in 0.5
do
for seq_len in 96
do
for label_len in 0
do
for pred_len in 0
do
for d_model in 768
do
for n_heads in 4
do
for e_layers in 4
do
for llm_layers in 3
do
for d_ff in 768
do
for bs in 128
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.001 
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u run_LLM4TS.py \
    --task_name imputation \
    --mask_rate $mask_rate \
    --lradj type1 \
    --random_seed 84 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --text_token_len 0 \
    --llm ./hf_models/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 1\
    --stride 1\
    --des $exp_des \
    --train_epochs 100\
    --patience 2\
    --target 'CHF' \
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --robust 0 \
    --blank 1 \
    --d_prompt 0 \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/imputation/ETTm1 \
    > logs/imputation/ETTm1/$mask_rate'_'$model_id'_sl'$seq_len'_pl'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_percent'$percent'_'$exp_des.log 2>&1 
done
done
done
done
done
done
done
done
done
done
done
done
done
done