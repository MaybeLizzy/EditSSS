#!/bin/bash

export target_model=gpt2-xl;  
export model_name=gpt2-xl-1.5B;
export editing_method=MEND;
export hparams_dir=$PWD/hparams/MEND/$target_model;
export data=zsre;
export data_dir=$PWD/data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json;
export metrics_save_dir=$PWD/output/portability/one_hop/$data/$editing_method;
export device=1;

Cs=(0.9);

for C in "${Cs[@]}"
do
    # FT
    # python test_one_hop.py --editing_method $editing_method --target_model $target_model --hparams_dir $hparams_dir --data $data --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    
    # MEND SERAC
    export archive="path to your trained model";
    python test_one_hop.py --editing_method $editing_method --target_model $target_model --archive $archive --hparams_dir $hparams_dir --data $data --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    
done