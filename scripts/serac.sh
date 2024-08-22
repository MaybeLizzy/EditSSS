#!/bin/bash


export editing_method=SERAC;
export target_model=gpt2-xl;
export model_name="path to your target model or you can download from huggingface";  
export cls_name=distilbert-base-cased;
export hparams_dir=$PWD/hparams/TRAINING/$editing_method/$target_model;

# zsre dataset
# export data=zsre;
# export train_path=$PWD/data/zsre/zsre_mend_train_10000.json;
# export eval_path=$PWD/data/zsre/zsre_mend_eval.json;
# export test_path=$PWD/data/zsre/zsre_mend_eval.json;

# counterfact dataset
export data=counterfact;
export train_path=$PWD/data/counterfact/counterfact-train.json;
export eval_path=$PWD/data/counterfact/counterfact-val.json;
export test_path=$PWD/data/counterfact/counterfact-edit.json;

export device=0;
export results_dir=$PWD/output/$editing_method/$target_model/$data/C=0.9;

python train.py --editing_method $editing_method --eval_size $eval_size --model_name ${model_name} --results_dir ${results_dir} --data $data --train_path ${train_path} --eval_path ${eval_path} --hparams_dir ${hparams_dir} --cls_name ${cls_name} --device $device; 

export test_hparams_dir=$PWD/hparams/$editing_method/$target_model;
export archive="path to your trained model";

python run_eval.py --editing_method $editing_method --target_model $target_model --data $data --model_name $model_name --hparams_dir $test_hparams_dir --archive $archive --data_dir $test_path --metrics_save_dir $results_dir --device $device;
