
import deepspeed
import torch
from transformers.models.llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import wptq_dev

def main():
    tokenizer=LlamaTokenizer.from_pretrained("/nvme/models/llama-2-13B")
    model=LlamaForCausalLM.from_pretrained(
        "/nvme/models/llama-2-13B",
        load_in_8bit=None ,
        device_map=None ,
        use_cache=False,
    )

    engine=deepspeed.init_inference(
        model=model,
        mp_size=2,
        dtype=torch.bfloat16,
    )

    # start to use wptq tool to quantize the model
    wptq_dev.utils.model_parse(model,quant_ops_type='qnet',model_name='llama2-13B')

    evaluate=wptq_dev.eval.Evaluator(tokenizer,ddp=True)

    # support dataset: mmlu,lambada, samsum
    evaluate.evaluate(model,model_name="llama-2-13b",dataset=['lambada'])

    wptq_dev.utils.model_celibration(model)
    
    wptq_dev.quant.smooth.smooth_lm(model,name='llama')

    evaluate.evaluate(model,model_name="llama-2-13b-quant-int8",dataset=['lambada'])


if __name__=="__main__":
    main()