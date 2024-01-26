# WPTQ: WITIN Post Train Quantization
## abstract
    welcome to WPTQ prejcet, the goal of the WPTQ is help users to run their custom networks on our CIM chip. our preject divided into 3 sub-prejects: Qnet,Onet and OnetIntInfer.
    Qnet: evaluate the efficacy of the quantization method
    Onet: simulate fully quantized network int inference by quant/dequant used on our chip 
    OnetIntInfer: simulate fully quantized network int inference by used on our chip,remove quant and dequant process

## model zoom 
|net| bit width |dataset| acc||
|:---:|:---:|:----:|:----:|:----:|
|ViT| fp32 | ImageNet | 80.33| base
|ViT| int8 | ImageNet | 75.58| except for gelu
|opt-13b| fp16 | lambada| 78.70| base
|opt-13b| int8 | lambada| 79.00| qnet
|llama2-13B| fp16  | lambada| 89.40| base
|llama2-13B| int8  | lambada| 89.09| qnet

## support ops

|ops| int infer type | source ops|
|:---:|:---:|:---:|
|Conv2d| qnet/onet|nn.Conv2d|
|DDPIntLinear| qnet | Linear with DDP|
|DDPLinearAllreduce| qnet | Linear with DDP|
|Linear| qnet/onet/OIntInfer | Linear |
|LayerNorm| qnet/onet/OIntInfer | LayerNorm |
|Softmax| qnet/onet/OIntInfer | softmax|
|MatMul| qnet/onet/OIntInfer | A@B|
|GELU| qnet/onet/OIntInfer| F.gelu|
|ElemAdd| qnet/onet/OIntInfer | A+B|
|ElemRightShift| qnet/onet/OIntInfer|A/n|
|Act| qnet/onet/OIntInfer| quant/dequant|


## How to use
prepare quant environment
``` base
    pip install -r requirement.txt
```
### use model zoo 
quant vit
``` base  
    bash run.sh "vit"
```
### use wptq tools to quant your custom network

### step 1:
add act ops to quant net intput and  dequant net output
prepare your network and replace target ops like follow:
``` python
    self.act1=QuantAct("quant_inputs")
    # your own network
    self.net = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
    self.act2=QuantAct("dequant_outputs")
``` 
### step 2:
replace target ops with our quant ops like follow
``` python
    # source,your network
    import wptq
    self.act1=wptq.nn.Act("quant_inputs")
    self.act2=wptq.nn.Act("dequant_outputs")
    self.linear1=wptq.nn.Linear(10, 20)
    self.linear2=wptq.nn.Linear(20, 10)
    self.soft=wptq.nn.Softmax()
    
    def forward(x):
        x=self.act1(x)
        x=self.linear1(x)
        x=self.linear2(x)
        ...
        y=self.soft(x)
        y=self.act2(y)
        return y
``` 
or replace target ops with config/config.json

```json
# replace each layer ops in your network
"re_ops":{
        "layers": 12,
        "ops":  
            {
                "vit.encoder.layer.id.attention.attention.query":
                {
                    "input":{
                        "quant_strategy":null,
                        "quant_bit":"int-8",
                        "quant_policy":"per-layer"
                    },
                    "output":{
                        "quant_strategy":null,
                        "quant_bit":"int-8",
                        "quant_policy":"per-layer"
                    },
                    "bias":{
                        "quant_strategy":null,
                        "quant_bit":"int-16",
                        "quant_policy":"per-layer"
                    },
                    "weight":{
                        "quant_strategy":null,
                        "quant_bit":"int-8",
                        "quant_policy":"per-layer"
                    }
                },
            }
        }
# replace all linear ops in your network
"on_ops": {
        "ops":["linear"]
    }
```
``` python
 wptq_dev.utils.model_parse(model,quant_ops_type='qnet',model_name='network')
```
### step 3:
start to quantization and evaluate and export onnx file
```python
    wptq.utils.model_celibretion(model) 
   
    fake_x=torch.randn(64,3,224,224).to("cuda")

    wptq_dev.utils.export_infer_net(trainer.model)

    trainer.model(fake_x)

    onnx_model=trainer.model.vit
    wptq_dev.utils.export_onnx_net(
        onnx_model,
        ".",
        (1,197,768),
    )

```


