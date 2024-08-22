#!/bin/bash


export editing_method=FT_SSS;
export target_model=gpt2-xl;   
export model_name="path to your target model or you can download from huggingface";
export hparams_dir=$PWD/hparams/$editing_method/$target_model;

# zsre dataset
export data=zsre;
export data_dir=$PWD/data/zsre/zsre_mend_eval.json;

# counterfact dataset
# export data=counterfact;
# export data_dir=$PWD/data/counterfact/counterfact-edit.json;
export device=0;

Cs=(0.9);

for C in "${Cs[@]}"
do
    export metrics_save_dir=$PWD/output/$editing_method/$target_model/$data/C=${C};

    python run_eval.py --editing_method $editing_method --target_model $target_model --hparams_dir $hparams_dir --data_dir $data_dir --data $data --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;

done
