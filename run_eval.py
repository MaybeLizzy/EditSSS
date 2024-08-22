import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset
from typing import List
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default='FT', type=str)
    parser.add_argument('--target_model', default='gpt2-xl', type=str)
    parser.add_argument('--model_name', default='gpt2-xl-1.5B', type=str)
    parser.add_argument('--hparams_dir', default='hparams/FT/gpt2-xl.yaml', type=str)
    parser.add_argument('--data_dir', default='data/zsre/zsre_mend_eval.json', type=str)
    parser.add_argument('--data', default='zsre', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='output', type=str)
    # parser.add_argument('--layer', default=None, type=int)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--archive', default=None, type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--keep_original_weight', default=1, type=int)
    parser.add_argument('--sequences', default=1, type=int)

    args = parser.parse_args()

    if args.editing_method.startswith('FT'):
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method.startswith('MEND'):
        editing_hparams = MENDHyperParams
    elif args.editing_method.startswith('SERAC'):
        editing_hparams = SERACHparams
    else:
        raise NotImplementedError

    test_data = json.load(open(args.data_dir, 'r', encoding='utf-8'))
    test_data = test_data[:10000]
    if args.keep_original_weight == 0:
        keep_original_weight = False
        test_data = test_data[:args.sequences]
        print("sequential edit!")
    else:
        keep_original_weight = True
    print("data: ", args.data)
    print("test data length: ", len(test_data))

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = []
    rephrase_prompts = [] 
    target_new = [] 
    locality_prompts = [] 
    locality_ans = [] 
    portability_prompts = []
    portability_ans = []
    subject = []
    if args.data == "zsre":
        for edit_data_ in test_data:
            if edit_data_['alt'] == "":
                continue
            else:
                prompts.append(edit_data_['src'])
                rephrase_prompts.append(edit_data_['rephrase'])
                target_new.append(edit_data_['alt'])
                locality_prompts.append(edit_data_['loc'])
                locality_ans.append(edit_data_['loc_ans'])
                subject.append(edit_data_['subject'])
                if 'portability' in edit_data_.keys():
                    portability_prompts.append(edit_data_['portability']['New Question'])
                    portability_ans.append(edit_data_['portability']['New Answer'])
    elif args.data == "counterfact":
        for edit_data_ in test_data:
            if edit_data_['target_new'] == "":
                continue
            else:
                prompts.append(edit_data_['prompt'])
                rephrase_prompts.append(edit_data_['rephrase_prompt'])
                target_new.append(edit_data_['target_new'])
                locality_prompts.append(edit_data_['locality_prompt'])
                locality_ans.append(edit_data_['locality_ground_truth'])
                subject.append(edit_data_['subject'])
                if 'portability' in test_data[0].keys():
                    portability_prompts.append(edit_data_['portability']['New Question'])
                    portability_ans.append(edit_data_['portability']['New Answer'])
    assert len(prompts) == len(rephrase_prompts) == len(target_new) == len(locality_prompts) == len(locality_ans) == len(subject)
    assert len(portability_prompts) == len(portability_ans)
    if len(portability_prompts) != 0:
        portability_inputs = {
            'one_hop':{
                'prompt': portability_prompts,
                'ground_truth': portability_ans
            },
        }
    else:
        portability_inputs = None

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }    
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    
    hparams.device = args.device
    if args.C is not None:
        hparams.C = args.C  
    if args.archive is not None:
        hparams.archive = args.archive
    if args.model_name is not None:
        hparams.model_name = args.model_name
    if args.editing_method == "FT_eigen":
        hparams.alg_name = args.editing_method

    if args.editing_method == 'IKE':
        train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = BaseEditor.from_hparams(hparams)    
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=keep_original_weight
    )
    
    # save edited model parameters
    save_dir = args.metrics_save_dir
    os.makedirs(save_dir, exist_ok=True)  
    if keep_original_weight is False:
        json.dump(metrics, open(os.path.join(save_dir, f'{args.editing_method}_{args.target_model}_C={args.C}_seq={args.sequences}_results.json'), 'w'), indent=4)
    else:
        json.dump(metrics, open(os.path.join(save_dir, f'{args.editing_method}_{args.target_model}_C={args.C}_results.json'), 'w'), indent=4)
    print("already dump json")