#!/bin/bash


export target_model=gpt2-xl; 
export model_name=gpt2-xl-1.5B;
export editing_method=FT_SSS;
export hparams_dir=$PWD/hparams/$editing_method/$target_model;
export data_dir=$PWD/data/locality/Distracting_Neighbor/counterfact_distracting_neighbor.json;
export metrics_save_dir=$PWD/output/locality/$target_model/$editing_method/layer0;
export device=0;

Cs=(0.9);

for C in "${Cs[@]}"
do
    # FT
    python test_locality.py --editing_method $editing_method --target_model $target_model --hparams_dir $hparams_dir --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    # MEND
    # export archive="path to your trained model";
    # python test_locality.py --editing_method $editing_method --target_model $target_model --archive $archive --hparams_dir $hparams_dir --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    # SERAC
    # export archive="path to your trained model";
    # python test_locality.py --editing_method $editing_method --target_model $target_model --archive $archive --hparams_dir $hparams_dir --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;

done