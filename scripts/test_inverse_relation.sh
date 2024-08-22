#!/bin/bash


export target_model=gpt2-xl;
export model_name=gpt2-xl-1.5B;
export editing_method=FT;
export hparams_dir=$PWD/hparams/MEND/$target_model;
export data_dir=$PWD/data/portability/Inverse_Relation/zsre_inverse_relation.json;
export metrics_save_dir=$PWD/output/portability/inverse_relation/$editing_method;
export device=0;

Cs=(0.9);

for C in "${Cs[@]}"
do
    # FT
    # python test_inverse_relation.py --editing_method $editing_method --target_model $target_model --hparams_dir $hparams_dir --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    # MEDN SERAC
    export archive="path to your trained model";
    python test_inverse_relation.py --editing_method $editing_method --target_model $target_model --archive $archive --hparams_dir $hparams_dir --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;

done