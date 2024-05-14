import argparse
import os
import torch.distributed as dist
import torch
from exp.exp_LLM4TS import Exp_Main
from exp.exp_imputation import Exp_Imputation
from exp.exp_ad import Exp_Anomaly_Detection
from exp.exp_sf import Exp_Short_Term_Forecast
from exp.exp_classification import Exp_Classification
import random
import numpy as np

parser = argparse.ArgumentParser(description='Time Series Forecasting')

parser.add_argument('--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# aLLMTS 
parser.add_argument('--is_llm', type=int, default=0, help='whether to use llm')
parser.add_argument('--pretrain', type=int, default=1, help='whether to use pretrained llm')
parser.add_argument('--freeze', type=int, default=1, help='whether to freeze specific part of the llm')
parser.add_argument('--llm_layers', type=int, default=1, help='the number of llm layers we use')
parser.add_argument('--mask_pt', type=int, default=0, help='mask pratrain ratio')
parser.add_argument('--llm', type=str, default='./HF_MODELS/gpt2', help='the llm checkpoint')
parser.add_argument('--attn_dropout', type=float, default=0, help='')
parser.add_argument('--proj_dropout', type=float, default=0, help='')
parser.add_argument('--res_attention', action='store_true', default=False, help='')

# sft
parser.add_argument('--sft', type=int, default=0, help='whether sft')
parser.add_argument('--sft_layers', type=str, default='null', help='the layers in llm needed to be trained')
parser.add_argument('--history_len', type=int, default=0, help='look-back window length')
parser.add_argument('--fft', type=int, default=0, help='fft')
parser.add_argument('--rand_init', type=int, default=0, help='rand_init')
# pt
parser.add_argument('--c_pt', type=int, default=0, help='whether continue pretrain')
parser.add_argument('--pt_layers', type=str, default='null', help='the layers in llm needed to be trained')
parser.add_argument('--pt_data', type=str, default='null', help='the dataset used in pretrain, use _ to separate')
parser.add_argument('--pt_sft', type=int, default=0, help='whether continue pretrain')
parser.add_argument('--pt_sft_base_dir', type=str, default='null', help='the base model dir for pt_sft')
parser.add_argument('--pt_sft_model', type=str, default='null', help='the base model for pt_sft')

# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--notrans', action='store_true', default=False, help='stop using transformer')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


if __name__ == '__main__':


    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)

    


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")

                

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Main
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Main
    

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_sl{}_pl{}_llml{}_lr{}_bs{}_percent{}_{}_{}'.format(
                args.model_id,
                args.seq_len,
                args.pred_len,
                args.llm_layers,
                args.learning_rate,
                args.batch_size,
                args.percent,
                args.des,
                ii,
                )


            exp = Exp(args) 
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.use_multi_gpu and args.use_gpu and args.local_rank != 0:
                pass
            else:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_sl{}_pl{}_llml{}_lr{}_bs{}_percent{}_{}_{}'.format(
                args.model_id,
                args.seq_len,
                args.pred_len,
                args.llm_layers,
                args.learning_rate,
                args.batch_size,
                args.percent,
                args.des,
                ii,
                )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
