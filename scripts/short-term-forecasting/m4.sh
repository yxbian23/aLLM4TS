export CUDA_VISIBLE_DEVICES=7

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/short_forecasting" ]; then
    mkdir ./logs/short_forecasting
fi

if [ ! -d "./checkpoints/short_forecasting" ]; then
    mkdir ./checkpoints/short_forecasting
fi

if [ ! -d "./logs/short_forecasting/m4" ]; then
    mkdir ./logs/short_forecasting/m4
fi

if [ ! -d "./checkpoints/short_forecasting/m4" ]; then
    mkdir ./checkpoints/short_forecasting/m4
fi



model_name=LLM4TS_sf


for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.002
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Monthly' \
    --model_id m4_Monthly \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_ff 128 \
    --d_model 128 \
    --patch_len 1 \
    --stride 1 \
    --batch_size $bs \
    --itr 1 \
    --learning_rate $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Monthly_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done

for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.005
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --model_id m4_Yearly \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --llm_layers 6 \
    --d_model 768 \
    --d_ff 32 \
    --patch_len 1 \
    --stride 1 \
    --batch_size $bs \
    --des 'Exp' \
    --itr 1 \
    --learning_rate $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Yearly_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done

for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.005
do
for sft_layers in ln_wpe 
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Quarterly' \
    --model_id m4_Quarterly \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --llm_layers 6 \
    --d_model 768 \
    --patch_len 1 \
    --stride 1 \
    --d_ff 128 \
    --batch_size  $bs \
    --des 'Exp' \
    --itr 1 \
    --learning_rate  $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Quarterly_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done

for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.001
do
for sft_layers in ln_wpe_attn_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Daily' \
    --model_id m4_Daily \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --llm_layers 6 \
    --d_model 768 \
    --d_ff 128 \
    --patch_len 1 \
    --stride 1 \
    --batch_size $bs \
    --des 'Exp' \
    --itr 1 \
    --learning_rate $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Daily_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done

for bs in 128
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.005
do
for sft_layers in ln_wpe_mlp
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Weekly' \
    --model_id m4_Weekly \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --llm_layers 6 \
    --d_model 768 \
    --d_ff 128 \
    --patch_len 1 \
    --stride 1 \
    --batch_size $bs \
    --des 'Exp' \
    --itr 1 \
    --learning_rate $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Weekly_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done

for bs in 16
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.005
do
for sft_layers in ln_wpe
do
    exp_des=$sft_layers'_'$pt_sft_model
    python -u run_LLM4TS.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Hourly' \
    --model_id m4_Hourly \
    --model $model_name \
    --data m4 \
    --lradj type1 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --llm_layers 6 \
    --d_model 768 \
    --d_ff 128 \
    --patch_len 1 \
    --stride 1 \
    --batch_size $bs \
    --des 'Exp' \
    --itr 1 \
    --learning_rate $lr \
    --loss 'SMAPE' \
    --random_seed 2021 \
    --percent $percent \
    --features M \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm ./hf_models/gpt2 \
    --des '' \
    --train_epochs 100\
    --patience 3\
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/short_forecasting/m4 \
    > logs/short_forecasting/m4/'m4_Hourly_llml_'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done