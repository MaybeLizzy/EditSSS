from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset, SERACTrainingHparams, KETrainingHparams, CounterFactDataset
import argparse
import os

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--editing_method', type=str, default='MEND_eigen_new')
parser.add_argument('--hparams_dir', type=str, default='hparams/TRAINING/MEND/gpt2-xl.yaml')
parser.add_argument('--model_name', type=str, default='gpt2-xl-1.5B')
parser.add_argument('--cls_name', type=str, default='distilbert-base-cased')
parser.add_argument('--results_dir', type=str, default='output/MEND')
parser.add_argument('--data', default='counterfact', type=str)
parser.add_argument('--train_path', type=str, default='counterfact-train.json')
parser.add_argument('--eval_path', type=str, default='counterfact-val.json')
parser.add_argument('--train_size', default=None, type=int)
parser.add_argument('--eval_size', default=None, type=int)
parser.add_argument('--max_iters', default=None, type=int)
parser.add_argument('--model_save_pt', default=None, type=int)
parser.add_argument('--val_interval', default=None, type=int)
parser.add_argument('--C', default=None, type=float)  # must be None
parser.add_argument('--device', default="0", type=int)
args = vars(parser.parse_args())

if args["editing_method"].startswith("MEND"):
    training_hparams = MENDTrainingHparams.from_hparams(args["hparams_dir"])
if args["editing_method"].startswith("SERAC"):
    training_hparams = SERACTrainingHparams.from_hparams(args["hparams_dir"])
if args["editing_method"].startswith("KE"):
    training_hparams = KETrainingHparams.from_hparams(args["hparams_dir"])

training_hparams.alg = args["editing_method"]
training_hparams.C = args["C"]
training_hparams.model_name = args["model_name"]
training_hparams.tokenizer_name = args["model_name"]
training_hparams.results_dir = args["results_dir"]
if args["editing_method"].startswith("SERAC"):
    training_hparams.small_name = args["model_name"]
    training_hparams.cls_name = args["cls_name"]    
    
if args["max_iters"] is not None:
    training_hparams.max_iters = args["max_iters"]
if args["model_save_pt"] is not None:
    training_hparams.model_save_pt = args["model_save_pt"]
if args["val_interval"] is not None:
    training_hparams.val_interval = args["val_interval"]
if args["device"] is not None:
    training_hparams.device = args["device"]

if args["data"] == "zsre":
    train_ds = ZsreDataset(args["train_path"], size=args["train_size"], config=training_hparams)
    eval_ds = ZsreDataset(args["eval_path"], size=args["eval_size"], config=training_hparams)
elif args["data"] == "counterfact":
    train_ds = CounterFactDataset(args["train_path"], size=args["train_size"], config=training_hparams)
    eval_ds = CounterFactDataset(args["eval_path"], size=args["eval_size"], config=training_hparams)

trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()