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
    parser.add_argument('--model_name', default='gpt2-xl', type=str)
    parser.add_argument('--hparams_dir', default='hparams/FT/gpt2-xl.yaml', type=str)
    parser.add_argument('--data_dir', default='data/locality/Distracting Neighbor/counterfact_distracting_neighbor.json', type=str)
    parser.add_argument('--data', default='counterfact', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='/output/locality', type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--archive', default=None, type=str)
    parser.add_argument('--device', default=1, type=int)

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
    generation_prompts = [] 
    target_new = [] 
    target_true = [] 
    portability_prompts = []
    portability_ans = []
    subject = []  
    unrelated_relation = []
    unrelated_relation_ans = []
    distracting_neighborhood_prompts = [] 
    attribute_prompts = []
    neighborhood_prompts =[]
    for edit_data_ in test_data:        
        prompt = edit_data_['requested_rewrite']['prompt']
        target_n = edit_data_['requested_rewrite']['target_new']["str"]
        target_t = edit_data_['requested_rewrite']['target_true']["str"]
        subj = edit_data_['requested_rewrite']['subject']
        prompt = prompt.replace("{}", subj)
        # prompt = prompt + " " + target_n
        prompts.append(prompt)
        generation_prompts.append(edit_data_['generation_prompts'])
        target_new.append(target_n)
        target_true.append(target_t)
        subject.append(subj)        
        random.seed(edit_data_["case_id"]) # fix random seed for reproduction
        unrelated_relation.append(edit_data_['unrelated_relation']["question"])
        unrelated_relation_ans.append(edit_data_['unrelated_relation']["object"])
        distracting_neighborhood_prompts.append(edit_data_['distracting_neighborhood_prompts'])
        attribute_prompts.append(edit_data_['attribute_prompts'])
        neighborhood_prompts.append(edit_data_['neighborhood_prompts'])
        if 'portability' in test_data[0].keys():
            portability_prompts.append(edit_data_['portability']['New Question'])
            portability_ans.append(edit_data_['portability']['New Answer'])
    
    assert len(prompts) == len(unrelated_relation) == len(target_new) == len(distracting_neighborhood_prompts) == len(attribute_prompts) == len(subject)
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
            'prompt': neighborhood_prompts,
            'ground_truth': target_true
        },
        'distracting_neighborhood':{
            'prompt': distracting_neighborhood_prompts,
            'ground_truth': target_true
        },
        'unrelated_relation':{
            'prompt': unrelated_relation,
            'ground_truth': unrelated_relation_ans
        },
        'attribute':{
            'prompt': attribute_prompts,
            'ground_truth': target_new
        },
        
    }    
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    
    hparams.device = args.device
    if args.batch_size is not None:
        hparams.batch_size = args.batch_size
        print("batch edit :",hparams.batch_size)
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
    if hparams.batch_size == 1:   
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=generation_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True
        )
    elif hparams.batch_size > 1:
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts,
            rephrase_prompts=generation_prompts,
            target_new=target_new,
            subject=subject,
            train_ds=train_ds,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True
        )
    else:
        raise ValueError
    
    # save edited model parameters
    save_dir = args.metrics_save_dir
    os.makedirs(save_dir, exist_ok=True)  
    if args.batch_size is not None:
        json.dump(metrics, open(os.path.join(save_dir, f'{args.editing_method}_{args.target_model}_C={args.C}_results_batch={args.batch_size}.json'), 'w'), indent=4)
    else:
        json.dump(metrics, open(os.path.join(save_dir, f'{args.editing_method}_{args.target_model}_C={args.C}_results.json'), 'w'), indent=4)
    print("already dump json")