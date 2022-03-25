#TODO: remove accelerator
# Code adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
import random
import torch
import hydra
import math
import json
import numpy as np
import os.path as osp

from functools import partial
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from glob import glob
from torch.utils.data import Dataset
from influence_utils.nn_utils import compute_influences

import transformers
#from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast
from transformers import (
    set_seed,
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling
)

import ipdb
import logging
logger = logging.getLogger(__name__)


def load_target_words(config):
    cont = open(osp.join(config.data_dir, "targets.txt")).read()
    target_words = cont.split("\n")
    target_words.pop(-1)
    
    cont = open(osp.join(config.data_dir, "truth/binary.txt")).read()
    binary = cont.split("\n")
    binary.pop(-1)
    binary = [i.split('\t')[1] for i in binary]
    
    cont = open(osp.join(config.data_dir, "truth/graded.txt")).read()
    grades = cont.split("\n")
    grades.pop(-1)
    grades = [i.split('\t')[1] for i in grades]
    
    return target_words, binary, grades
    
    
def filter_fn(example, idx, remove_target, corpus_name=None, target_word=None, fast_tokenizer=None):
    """_summary_ 
    return True for unwanted examples

    Args:
        remove_or_keep (_type_): _description_
        corpus_name (_type_, optional): corpus to keep. Defaults to None.
    Returns:
        _type_: _description_
    """
    #contain_target_word = target_word in fast_tokenizer.decode(example["input_ids"])
    contain_target_word = target_word in example["text"]
    
    if remove_target:
        return not contain_target_word if corpus_name is None else (not contain_target_word) and example["source"] == corpus_name
    else:
        return contain_target_word if corpus_name is None else contain_target_word and example["source"] == corpus_name


def cal_size(model):
    return sum([np.prod(p.size()) for p in model.parameters()])


def load_fast_tokenizer(config):
    
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    
    tokenizer = ByteLevelBPETokenizer(
        osp.join(config.snapshot_dir, "tokenizer-vocab.json"), 
        osp.join(config.snapshot_dir, "tokenizer-merges.txt")
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    tokenizer.enable_truncation(max_length=config.max_length)


    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.mask_token = "<mask>"
    fast_tokenizer.unk_token = "<unk>"
    
    return fast_tokenizer
    
    
def get_latest_weights(config):
    
    all_weight_paths = list(Path(config.snapshot_dir).glob("model-*"))
    all_weight_paths = sorted(all_weight_paths, key=lambda k: int(str(k.name).replace("model-", "")))
    return all_weight_paths[-1]

    
@hydra.main(config_path="./config", config_name="eval_bert")
def main(config):

    #accelerator = init_accelerator()
    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)
   
    
    raw_datasets = load_dataset("text", 
        data_files={
            "train": osp.join(config.data_dir, "full_corpus/train/train.txt"), 
            "val": osp.join(config.data_dir, "full_corpus/val/val.txt"), 
        }
    )
    
    fast_tokenizer = load_fast_tokenizer(config)
    
    
    logger.info(f"Load the checkpoint from {get_latest_weights(config)}")
    model = AutoModelForMaskedLM.from_pretrained(get_latest_weights(config))
    

    logger.info(f"Model size: {cal_size(model)}")   # It was 89887880 (89M)
    
    ### Pre-process
    def tokenize_function(examples):
        
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        tokenized_examples = fast_tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        tokenized_examples.update({"text": examples["text"]})
        return tokenized_examples
        
    #with accelerator.main_process_first():
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line"
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["val"]
    
    
    
    train_source = open(osp.join(config.data_dir, "full_corpus/train/train_source.txt")).read().split("\n")
    train_source.pop()
    
    eval_source = open(osp.join(config.data_dir, "full_corpus/val/val_source.txt")).read().split("\n")
    eval_source.pop()
    
    train_dataset = train_dataset.add_column("source", train_source)
    eval_dataset = eval_dataset.add_column("source", eval_source)
    
    # load target words
    target_words, binary_gts, grade_gts = load_target_words(config)

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=fast_tokenizer, 
        mlm_probability=config.mlm_probability
    )

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = {n: 0.0 if any(nd in n for nd in no_decay) else config.weight_decay for n, p in model.named_parameters()}
    
    out_filename = f"scale_{config.hessian_approx.scale}-recursion_depth_{config.hessian_approx.recursion_depth}"
    if config.normalize_inf:
        out_filename += "-normalize_inf"

    out_filename += ".json"
    out_filename = osp.join(config.snapshot_dir, out_filename)
    if osp.exists(out_filename):
        save_outputs = json.load(open(out_filename))
    else:
        save_outputs = []

    for target_word, binary_gt, grade_gt in zip(target_words, binary_gts, grade_gts):
        logger.info(f"***** {target_word}: {binary_gt}, {grade_gt} *****")
        
        for corpus_name in ["ccoha1.txt", "ccoha2.txt"]:

            match = np.array([i[0] == target_word and i[1] == corpus_name for i in save_outputs])        
            if any(match):
                i = np.where(match)[0][0]
                influences = save_outputs[i][2]
            else:
                logger.info(f"Remove sentences containing `{target_word}` from {corpus_name}")
                train_corpus_1_targets_dataset = train_dataset.filter(
                    partial(filter_fn, 
                        remove_target=False, 
                        corpus_name=corpus_name, 
                        target_word=target_word, 
                        fast_tokenizer=fast_tokenizer
                    ), 
                    with_indices=True, 
                    load_from_cache_file=False, 
                    num_proc=config.num_proc,
                    desc=f"Filter sentences for {target_word}"
                )
                
                eval_targets_dataset = eval_dataset.filter(
                    partial(filter_fn, 
                        remove_target=False, 
                        target_word=target_word, 
                        fast_tokenizer=fast_tokenizer
                    ), 
                    with_indices=True, 
                    load_from_cache_file=False, 
                    num_proc=config.num_proc,
                    desc=f"Filter sentences for {target_word}"
                )
                
                
                logger.info(f"Train set size {len(train_corpus_1_targets_dataset)}, eval set size: {len(eval_targets_dataset)}")
                influences = compute_influences(
                    config, 
                    model, 
                    train_corpus_1_targets_dataset.remove_columns(["text", "source"]), 
                    eval_targets_dataset.remove_columns(["text", "source"]), 
                    data_collator, 
                    optimizer_grouped_parameters
                )
                influences = influences.item()
                
                save_outputs.append([target_word, corpus_name, influences, binary_gt, grade_gt, len(train_corpus_1_targets_dataset)])

            logger.info(f"Influence: {influences}")
            json.dump(save_outputs, open(out_filename, "w"))
            
            
if __name__ == "__main__":
    main()
