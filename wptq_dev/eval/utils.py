from datasets import load_dataset
from transformers.data import DataCollatorForSeq2Seq
from llama_recipes.data.sampler import LengthBasedBatchSampler
from llama_recipes.data.concatenator import ConcatDataset
import torch
from transformers import default_data_collator
samsum_path   = "/nvme/dataset/samsum/"
lambada_path  = "/home/wangh/code/llm_quant/data/lambada"

def _get_preprocessed_samsum(tokenizer, split):
    dataset = load_dataset(samsum_path, split=split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

def _get_preprocessed_lambada(tokenizer,split='validation'):
    dataset=load_dataset(lambada_path,split=split)
    
    def tokenize_function(examples):
        example = tokenizer(examples['text'])
        return example
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    dataset.set_format(type='torch', columns=['input_ids'])
    
    return dataset


def _prepare_dataloader(dataset,tokenizer,packing=True):
    val_dl_kwargs={}
    batch_size  =   1
    if packing:
        val_dl_kwargs["collate_fn"] = default_data_collator
        val_dl_kwargs["drop_last"] =True
        val_dl_kwargs["batch_size"] = batch_size   
    else:
        val_dl_kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        val_dl_kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=False)

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        pin_memory=True,
        **val_dl_kwargs,
    )   

    return eval_dataloader

def get_preprocessed(dataset,tokenizer,split):    
    if dataset=='lambada':
        dataset=_get_preprocessed_lambada(tokenizer,split)
        return dataset
    elif dataset=='samsum':
        packing=True
        def get_split():
            return (
                'train' if split=='train' else 'validation'
            )
        dataset=_get_preprocessed_samsum(tokenizer,get_split())
        if packing:
            dataset = ConcatDataset(dataset, chunk_size=4096)
        return _prepare_dataloader(dataset,tokenizer)
    else:
        raise ValueError