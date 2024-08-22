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
    parser.add_argument('--data_dir', default='data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json', type=str)
    parser.add_argument('--data', default='zsre', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='output/portability/one_hop', type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--archive', default=None, type=str)
    parser.add_argument('--device', default=0, type=int)

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
    elif args.editing_method.startswith('ROME'):
        editing_hparams = ROMEHyperParams
    elif args.editing_method.startswith('MEMIT'):
        editing_hparams = MEMITHyperParams
    else:
        raise NotImplementedError

    test_data = json.load(open(args.data_dir, 'r', encoding='utf-8'))
    print("test data length: ", len(test_data))

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = []
    rephrase_prompts = [] 
    target_new = [] 
    portability = []  
    portability_answer = []
    subject = []
    if args.data == "zsre":
        for edit_data_ in test_data:
            prompts.append(edit_data_['src'])
            rephrase_prompts.append(edit_data_['rephrase'])
            target_new.append(edit_data_['alt'])
            portability.append(edit_data_["portability"]["New Question"])        
            portability_answer.append(edit_data_["portability"]["New Answer"])
            subject.append(edit_data_['subject'])
    elif args.data == "counterfact":
        for edit_data_ in test_data:
            pp = edit_data_['requested_rewrite']['prompt']
            target_n = edit_data_['requested_rewrite']['target_new']["str"]
            subj = edit_data_['requested_rewrite']['subject']
            prompt = pp.replace("{}", subj)
            prompts.append(prompt)
            target_new.append(target_n)
            portability.append(edit_data_["portability"]["New Question"])
            portability_answer.append(edit_data_["portability"]["New Answer"])
            rephrase_prompts.append(edit_data_['paraphrase_prompts'])
            subject.append(subj)
    assert len(prompts) == len(portability) == len(target_new) == len(portability_answer)
    
    portability_inputs = {
        'one_hop':{
            'prompt': portability,
            'ground_truth': portability_answer
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
        locality_inputs=None,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )
    
    # save edited model parameters
    save_dir = args.metrics_save_dir
    os.makedirs(save_dir, exist_ok=True)  
    json.dump(metrics, open(os.path.join(save_dir, f'{args.editing_method}_{args.target_model}_C={args.C}_results.json'), 'w'), indent=4)
    print("already dump json")