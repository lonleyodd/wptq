import torch
from transformers import GPT2Tokenizer
from transformers.models.opt import OPTForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import wptq_dev
    
class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset 
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])
        self.device = 'cuda'
        self.eval_dataloader=None
        

    @torch.no_grad()
    def acc_eval(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc 
    
    def ppl_eval(self,model):
        for step, batch in enumerate(tqdm(self.eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        eval_epoch_loss = eval_loss / len(self.eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        return eval_ppl

    def evaluate(self,model,metric='acc'):
        acc=None
        ppl=None
        if metric=='acc':
            acc=self.acc_eval(model)
        if metric=='ppl':
            ppl=self.ppl_eval(model)
        eval_info_cfg={
            'samsum':{
                'ppl':ppl
            },
            'lambada':{
                'acc':acc
            }
        }
        return eval_info_cfg



def main():
    model_path="/nvme/wangh/model/opt-13b"
    data_path="/home/wangh/code/llm_quant/data/lambada"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    dataset = load_dataset(data_path, split='validation[:100]')
    evaluator = Evaluator(dataset, tokenizer)
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')

    logger=wptq_dev.utils.model_parse(model,quant_ops_type='qnet',model_name='opt-13B')

    acc_ = evaluator.evaluate(model)
    logger.info(f'base model accuracy: {acc_}')
    
    wptq_dev.quant.smooth.smooth_lm(model,name='opt')
    wptq_dev.utils.model_celibration(model)
    
    acc_int8 = evaluator.evaluate(model)
    logger.info(f'int8 model (int8) accuracy: {acc_int8}')




if __name__=="__main__":
    main()