export CUDA_VISIBLE_DEVICES=6

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/anomaly_detection" ]; then
    mkdir ./logs/anomaly_detection
fi

if [ ! -d "./checkpoints/anomaly_detection" ]; then
    mkdir ./checkpoints/anomaly_detection
fi

if [ ! -d "./logs/anomaly_detection/PSM" ]; then
    mkdir ./logs/anomaly_detection/PSM
fi

if [ ! -d "./checkpoints/anomaly_detection/PSM" ]; then
    mkdir ./checkpoints/anomaly_detection/PSM
fi



model_name=LLM4TS_ad

root_path_name=./dataset/PSM
data_path_name=PSM.csv
data_name=PSM
model_id=$data_name'_'$model_name

for anomaly_ratio in 1
do
for seq_len in 100
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
for llm_layers in 6
do
for d_ff in 768
do
for bs in  128
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
    --task_name anomaly_detection \
    --anomaly_ratio $anomaly_ratio \
    --d_ff 768 \
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
    --enc_in 25 \
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
    --train_epochs 12\
    --patience 3\
    --target 'CHF' \
    --itr 1 \
    --batch_size $bs \
    --learning_rate $lr \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/test \
    --pt_sft_model $pt_sft_model \
    --sft 1 \
    --sft_layers $sft_layers \
    --checkpoints ./checkpoints/anomaly_detection/PSM \
    > logs/anomaly_detection/PSM/$anomaly_ratio'_'$model_id'_sl'$seq_len'_pl'$pred_len'_llml'$llm_layers'_lr'$lr'_bs'$bs'_percent'$percent'_'$exp_des.log 2>&1 
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