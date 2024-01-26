
import deepspeed
import torch
from transformers.models.llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from wptq_dev.eval import Evaluator
from wptq_dev.utils import model_parse,model_celibration
from wptq_dev.quant.smooth import smooth_lm

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
    model_parse(model)

    evaluate=Evaluator(tokenizer,ddp=True)

    evaluate.evaluate(model,dataset=['mmlu'],model_name="llama-2-13b")

    model_celibration(model)
    
    smooth_lm(model,name='llama')

    evaluate.evaluate(model,dataset=['mmlu'],model_name="llama-2-13b-quant-int8")


if __name__=="__main__":
    main()