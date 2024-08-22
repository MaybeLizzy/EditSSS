#!/bin/bash

export target_model=gpt2-xl;  
export model_name=gpt2-xl-1.5B;
export editing_method=SERAC_SSS;
export hparams_dir=$PWD/hparams/SERAC/$target_model;
export data=counterfact;
export data_dir=$PWD/data/portability/Subject_Replace/counterfact_subject_replace.json;
export metrics_save_dir=$PWD/output/portability/subject_replace/$data/$editing_method;
export device=0;

Cs=(0.9);

for C in "${Cs[@]}"
do
    # FT
    # python test_subject.py --editing_method $editing_method --target_model $target_model --hparams_dir $hparams_dir --data $data --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;
    
    # MEND SERAC
    export archive="path to your trained model";
    python test_subject.py --editing_method $editing_method --target_model $target_model --archive $archive --hparams_dir $hparams_dir --data $data --data_dir $data_dir --model_name $model_name --metrics_save_dir $metrics_save_dir --device $device --C $C;

done