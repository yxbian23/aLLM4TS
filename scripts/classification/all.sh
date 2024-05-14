export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/classification" ]; then
    mkdir ./logs/classification
fi

if [ ! -d "./checkpoints/classification" ]; then
    mkdir ./checkpoints/classification
fi



pt_sft_model=pt_ckpt
sft_layers=ln_wpe_mlp
model_name=LLM4TS_cls


data=EthanolConcentration

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8
do
for bs in 64
do
for lr in 0.002
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe_attn_mlp
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=FaceDetection

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 512 
do
for lr in 0.002 0.0005 
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=Handwriting

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 16
do
for lr in 0.002 0.0005
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=Heartbeat

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done


data=JapaneseVowels

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done



data=PEMS-SF

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 512
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done


data=SelfRegulationSCP1

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=SelfRegulationSCP2

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=SpokenArabicDigits

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done

data=UWaveGestureLibrary

if [ ! -d "./logs/classification/"$data ]; then
    mkdir ./logs/classification/$data
fi

if [ ! -d "./checkpoints/classification/"$data ]; then
    mkdir ./checkpoints/classification/$data
fi

for pt_sft_model in all_sl1024_6_lr0.0001_bs768_s8 
do
for bs in 128
do
for lr in 0.002 0.0005 0.0001
do
for patch in 16 8
do
for stride in 16 8 4 2
do
for sft_layers in ln_wpe 
do
python -u run_LLM4TS.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$data/ \
  --model_id $model_name'_'$data \
  --model $model_name \
  --data UEA \
  --is_llm 1 \
  --pretrain 1 \
  --freeze 1 \
  --llm_layers 6 \
  --llm ./hf_models/gpt2 \
  --d_model 768 \
  --d_ff 768 \
  --patch_len $patch\
  --stride $stride\
  --train_epochs 100\
  --patience 10\
  --itr 1 \
  --batch_size $bs \
  --learning_rate $lr \
  --random_seed 2021 \
  --pt_sft 1 \
  --pt_sft_base_dir ./checkpoints/pt_patch/test \
  --pt_sft_model $pt_sft_model \
  --sft 1 \
  --sft_layers $sft_layers \
  --checkpoints ./checkpoints/classification/$data \
  --des 'exp'\
  --lradj type1 \
  > logs/classification/$data/$model_name'_'$data'_p_'$patch'_s_'$stride'_lr'$lr'_bs'$bs'_'$sft_layers'_'$pt_sft_model.log 2>&1 
done
done
done
done
done
done
