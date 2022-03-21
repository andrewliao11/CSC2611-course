#TODO: remove accelerator
# Code adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py
import random
import torch
import hydra
import math
import numpy as np
import os.path as osp

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from glob import glob
from torch.utils.data import Dataset

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



def cal_size(model):
    return sum([np.prod(p.size()) for p in model.parameters()])


# TODO: Avoid trimming/splitting down target words
def get_fast_tokenizer(config):
    
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=list(glob(osp.join(config.data_dir, "full_corpus/train/*.txt"))), 
        vocab_size=config.vocab_size, 
        min_frequency=config.min_frequency, 
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",]
    )
    
    tokenizer_paths = tokenizer.save_model(".", "tokenizer")
    
    
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    
    tokenizer = ByteLevelBPETokenizer(*tokenizer_paths)
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

    
'''   
def init_accelerator():
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()
    return accelerator
  '''
    
    
@hydra.main(config_path="./config", config_name="train_bert")
def main(config):

    #accelerator = init_accelerator()

    if config.use_wandb:# and accelerator.is_local_main_process:
        import wandb
        wandb.init(
            project="csc2611-course-semantic-change", 
            name=f"vocab_size_{config.vocab_size}-batch_size{config.per_device_train_batch_size}-learning_rate_{config.learning_rate}", 
            entity="andrewliao11", 
            config=config
        )
            
    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)
        
    
    raw_datasets = load_dataset("text", 
        data_files={
            "train": osp.join(config.data_dir, "full_corpus/train/train.txt"), 
            "val": osp.join(config.data_dir, "full_corpus/val/val.txt"), 
        }
    )
    
    fast_tokenizer = get_fast_tokenizer(config)
    bert_auto_config = AutoConfig.from_pretrained("bert-base-uncased")
    bert_auto_config.hidden_size = 8*16         # original: 12 * 64
    bert_auto_config.num_attention_heads = 8    # original: 12
    
    model = AutoModelForMaskedLM.from_config(bert_auto_config)
    model.resize_token_embeddings(len(fast_tokenizer))
    

    logger.info(f"Model size: {cal_size(model)}")   # It was 89887880 (89M)
    
    ### Pre-process
    def tokenize_function(examples):
        
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return fast_tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        
    #with accelerator.main_process_first():
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset line_by_line",
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["val"]

    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    
    ### Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=fast_tokenizer, 
        mlm_probability=config.mlm_probability
    )
    
    
    ### DataLoaders creation
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=config.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=config.per_device_eval_batch_size
    )


    ### Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # Prepare everything with our `accelerator`.
    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #    model, optimizer, train_dataloader, eval_dataloader
    #)
    
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        
    lr_scheduler = transformers.get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_train_steps,
    )
    
    #if accelerator.is_main_process:
    fast_tokenizer.save_pretrained(".")
    
    
    # Train!
    total_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps# * accelerator.num_processes


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    # Only show the progress bar once on each machine.
    #progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(range(config.max_train_steps))
    completed_steps = 0

    
    model.cuda()
    
    for epoch in range(config.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / config.gradient_accumulation_steps
            
            
            #accelerator.backward(loss)
            loss.backward()
            
            if step % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if config.use_wandb:
                    wandb.log({
                        "step": completed_steps, 
                        "train/loss": loss
                    })

            if completed_steps >= config.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)

            loss = outputs.loss
            #losses.append(accelerator.gather(loss.repeat(config.per_device_eval_batch_size)))
            losses.append(loss.repeat(config.per_device_eval_batch_size))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        
        
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch, 
                "eval/loss": torch.mean(losses), 
                "eval/perplexity": perplexity
            })
            
        #accelerator.wait_for_everyone()
        #unwrapped_model = accelerator.unwrap_model(model)
        #unwrapped_model.save_pretrained(f"model-{epoch}", save_function=accelerator.save)
        model.save_pretrained(f"model-{epoch}")
        #to load: model = AutoModelForMaskedLM.from_pretrained("model")

    
if __name__ == "__main__":
    main()
