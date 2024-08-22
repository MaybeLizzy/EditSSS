#!/bin/bash

export editing_method=MEND;
export target_model=gpt2-xl;    
export model_name="path to your target model or you can download from huggingface";   
export results_dir=$PWD/output;
export hparams_dir=$PWD/hparams/TRAINING/$editing_method/$target_model;

# zsre dataset
# export data=zsre;
# export train_path=$PWD/data/zsre/zsre_mend_train_10000.json;
# export eval_path=$PWD/data/zsre/zsre_mend_eval.json;

# counterfact dataset
export data=counterfact;
export train_path=$PWD/data/counterfact/counterfact-train.json;
export eval_path=$PWD/data/counterfact/counterfact-val.json;
export test_path=$PWD/data/counterfact/counterfact-edit.json;

export device=1;

python train.py --editing_method $editing_method --hparams_dir ${hparams_dir} --data $data --model_name ${model_name} --results_dir ${results_dir} --train_path ${train_path} --eval_path ${eval_path} --device $device; 

export test_hparams_dir=$PWD/hparams/$editing_method/$target_model;
export archive="path to your trained model";
export results_dir="path to your result directory";


python run_eval.py --editing_method $editing_method --target_model $target_model --data $data --hparams_dir $test_hparams_dir --archive $archive --data_dir $test_path --model_name $model_name --metrics_save_dir $results_dir --device $device;

# sequential edit
# export keep_original_weight=0;
# sequences=(10 100 1000);
# for seq in "${sequences[@]}"
# do
#     export results_dir=$PWD/output/sequential/$editing_method/$target_model/$data;
#     python run_eval.py --editing_method $editing_method --target_model $target_model --keep_original_weight $keep_original_weight --sequences $seq --data $data --hparams_dir $test_hparams_dir --archive $archive --data_dir $test_path --model_name $model_name --metrics_save_dir $results_dir --device $device;
# done