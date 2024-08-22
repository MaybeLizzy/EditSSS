#!/bin/bash


export editing_method=MEND_SSS;
export target_model=gpt2-xl;
export model_name="path to your target model or you can download from huggingface";  
export results_dir=$PWD/output;
export hparams_dir=$PWD/hparams/TRAINING/MEND/$target_model;

# zsre dataset
# export data=zsre;
# export train_path=$PWD/data/zsre/zsre_mend_train_10000.json;
# export eval_path=$PWD/data/zsre/zsre_mend_eval_portability_gpt4.json;

# counterfact dataset
export data=counterfact;
export train_path=$PWD/data/counterfact/counterfact-train.json;
export eval_path=$PWD/data/counterfact/counterfact-val.json;
export test_path=$PWD/data/counterfact/counterfact-edit.json;

export device=1;
export test_hparams_dir=$PWD/hparams/MEND/$target_model;
export archive="path to your trained model";
export results_dir="path to your result directory";

Cs=(0.9);
for C in "${Cs[@]}"
do
    python train.py --editing_method $editing_method --hparams_dir ${hparams_dir} --model_name ${model_name} --data $data --results_dir ${results_dir} --train_path ${train_path} --eval_path ${eval_path} --C $C --device $device; 
 
    python run_eval.py --editing_method $editing_method --target_model $target_model --data $data --hparams_dir $test_hparams_dir --archive $archive --metrics_save_dir $results_dir --data_dir $test_path --model_name $model_name --C $C --device $device;

done
