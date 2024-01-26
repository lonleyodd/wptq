import torch
from tqdm import tqdm
from .mmlu import mmlu_eval
from .utils import get_preprocessed
import os

class Evaluator:
    def __init__(self, tokenizer,ddp=False):
        self.tokenizer = tokenizer
        self.device = 'cuda'
        self.ddp=ddp
        self.eval_info={}
    
    @torch.no_grad()
    def eval_lambada(self, model,):
        dataset=get_preprocessed('lambada',self.tokenizer,'validation[:100]')
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(dataset,colour='green',desc='evaluating acc',dynamic_ncols=True):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        self.eval_info.update(
            {'lambada':{
                'sample':str(len(dataset)),
                'ppl':'-',
                'loss':'-',
                'acc':str(round(acc,4))
            }
            }
        )

    def eval_samsum(self,model):
        val_loader=get_preprocessed('samsum',self.tokenizer,'test')
        eval_loss = 0.0
        for step, batch in enumerate(tqdm(val_loader,colour="green", desc="evaluating ppl", dynamic_ncols=True)):
            for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        eval_epoch_loss = eval_loss / len(val_loader)
        eval_ppl = torch.exp(eval_epoch_loss)
        self.eval_info.update(
            {
                'samsum':{
                    'sample':str(len(val_loader.dataset)),
                    'ppl':str(round(eval_ppl.item(),4)),
                    'loss':str(round(eval_epoch_loss.item(),4)),
                    'acc':'-'
                },
            }
        )

    def evaluate(self,model,model_name,dataset=['lambada','samsum','mmlu']):
        acc=None
        ppl=None
        
        if 'lambada' in dataset:
            self.eval_lambada(model)
        if 'samsum' in  dataset:
            self.eval_samsum(model)
        if 'mmlu' in dataset:
            mmlu_eval(model,self.tokenizer,model_name)
            
        if self.ddp==True and int(os.environ['LOCAL_RANK'])==0:
            print("| {:<10s} | {:<8s} | {:<8s} |{:<8s} |{:<8s} |".format("data", "sample", "ppl","loss","acc"))
            for data,info in self.eval_info.items():
                print("|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|"+ "-"*9 + "|"+ "-"*10+"|")    
                print("| {:<10s} | {:<8s} | {:<8s} | {:<8s} | {:<8s} |".format(data, info["sample"], info["ppl"],info["loss"],info["acc"]))
                    
        