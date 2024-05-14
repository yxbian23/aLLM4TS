export CUDA_VISIBLE_DEVICES=7

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/long-term-forecasting" ]; then
    mkdir ./logs/long-term-forecasting
fi

if [ ! -d "./checkpoints/long-term-forecasting" ]; then
    mkdir ./checkpoints/long-term-forecasting
fi

if [ ! -d "./logs/long-term-forecasting/illness" ]; then
    mkdir ./logs/long-term-forecasting/illness
fi

if [ ! -d "./checkpoints/long-term-forecasting/illness" ]; then
    mkdir ./checkpoints/long-term-forecasting/illness
fi



model_name=LLM4TS_sft_zero

root_path_name=./dataset/
data_path_name=illness.csv
data_name=custom
model_id=$data_name'_'$model_name


for d_model in 768
do
for llm_layers in 6
do
for d_ff in 768
do
for bs in 8
do
for percent in 100 
do
for pt_sft_model in pt_ckpt
do
for lr in 0.0005
do
for sft_layers in ln_wpe 
do
    exp_des=$sft_layers'_'$pt_sft_model'_gpt2'
    python -u run_LLM4TS.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id \
    --model $model_name \
    --data $data_name \
    --percent $percent \
    --features M \
    --seq_len 60 \
    --pred_len 60 \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers $llm_layers \
    --llm ./hf_models/gpt2 \
    --affine 1 \
    --enc_in 7 \
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des $exp_des \
    --train_epochs 100\
    --patience 10\
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --history_len 60 \
    --checkpoints ./checkpoints/long-term-forecasting/illness \
    > logs/long-term-forecasting/illness/$model_id'_336_720_llml'$llm_layers'_lr'$lr'_bs'$bs'_'$exp_des.log 2>&1 
done
done
done
done
done
done
done
done