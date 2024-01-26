import os
import torch

from typing import List
from .log import get_logger
from ..nn import *
from ..ddp_nn import DDPIntLinear,DDPLinearAllreduce
from transformers.activations import GELUActivation
import json
from itertools import product
from deepspeed.module_inject.layers import LinearLayer,LinearAllreduce


_quant_ops=(
    QuantElemRightShift,QuantLinear,
    QuantSoftmax, QuantLayerNorm, QuantMatMul,
    QuantGelu, QuantAct, QuantElemAdd, 
    QuantElemRightShift,DDPIntLinear,DDPLinearAllreduce
)
_model_zoom = {
    'vit': "/home/wangh/code/wptq/wptq_dev/model_zoo/vit.json",
    'opt-13B': "/home/wangh/code/wptq/wptq_dev/model_zoo/opt-13B.json",
    'llama2-13B': "/home/wangh/code/wptq/wptq_dev/model_zoo/llama2-13B.json",
}

_base_cfg_path = "/home/wangh/code/wptq/wptq_dev/config/ops.json"

def _ops_replace(model,module_name,ops):
    name_list=module_name.split('.')
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)  
    setattr(model, name_list[-1],ops)

def ops_replace(model, ops_name, ops_type, ops,quant_ops_type,quant_args):
    if ops_type == torch.nn.Softmax:
        layer_cls = QuantSoftmax(quant_ops_type) 
    elif ops_type == torch.nn.LayerNorm: 
        layer_cls = QuantLayerNorm(ops,quant_ops_type) 
    elif ops_type == torch.nn.Linear:
        layer_cls=QuantLinear(ops,quant_ops_type,**quant_args)
    elif ops_type == torch.nn.Conv2d:
        layer_cls=QuantConv2d(ops,quant_ops_type)
    elif ops_type == GELUActivation:
        layer_cls=QuantGelu(quant_ops_type) 
    # deepspeed
    # TODO ddp for onet ops
    elif ops_type == LinearLayer:
        layer_cls=DDPIntLinear(ops,quant_ops_type,**quant_args) 
    elif ops_type == LinearAllreduce:
        layer_cls=DDPLinearAllreduce(ops,quant_ops_type,**quant_args) 
    else:
        ValueError
    _ops_replace(model,ops_name,layer_cls)


def find_target_ops(cfg_path):
    with open(cfg_path,'r',encoding="utf-8") as f:
        data=json.loads(f.read())
    ops_dict=None
    ops_class=None
    if data['re_ops'] is not None:
        data1=data['re_ops']
        layers=data1['layers']
        _ops=data1['ops']
        if isinstance(layers,int):
            layers=[i for i in range(layers)]
            ops_dict = { o.replace('id',str(l)): _ops[o] for o,l in product(_ops.keys(),layers)}
        elif isinstance(layers,List):
            ops = [ o.replace('id',str(l)) for o,l in product(_ops,layers)]
        else:
            raise ValueError(f"invalid layers{layers} is re_ops")
    if data['on_ops'] is not None:
        data2=data['on_ops']
        _ops=data2['ops']
        if _ops is not None:
            ops_class_list=[]
            for op in _ops:
                if op.lower()=='linear':
                    ops_class_list.append(torch.nn.Linear)
                elif op.lower()=='softmax':
                    ops_class_list.append(torch.nn.Softmax)
                elif op.lower()=='layernorm':
                    ops_class_list.append(torch.nn.LayerNorm)
                elif op.lower()=='conv2d':
                    ops_class_list.append(torch.nn.Conv2d)
                elif op.lower()=='gelu':
                    ops_class_list.append(GELUActivation)
                else:
                    raise ValueError(f'error in ops {op}')
            ops_class=tuple(ops_class_list)
    return ops_dict,ops_class

def model_parse(model:torch.nn.Module,quant_ops_type='qnet',model_name=None):
    if model_name in _model_zoom.keys():
        cfg_path=_model_zoom[model_name]
    else:
        cfg_path=_base_cfg_path
    ops_dict,ops_tuple=find_target_ops(cfg_path)
    
    if ops_dict is None and ops_tuple is None:
        return
    
    for n,m in model.named_modules():
        if (ops_dict is not None and n in ops_dict.keys()) :
            quant_args=ops_dict[n] if ops_dict is not None else None
            ops_replace(model,n,type(m),m,quant_ops_type,quant_args)
        if (ops_tuple is not None and isinstance(m,ops_tuple)):
            quant_args=None
            ops_replace(model,n,type(m),m,quant_ops_type,quant_args)
            
    return register_hook(model)

        
def register_hook(model):
    hook=Hooks()
    logger=get_logger()
    for m,n in model.named_modules():
        if isinstance(n,_quant_ops):
            setattr(n,"ops_id",hook.id)
            setattr(n,"module_name",m)
            setattr(n,"logger",logger)
    return logger

def model_celibration(model:torch.nn.Module):
    for m,n in model.named_modules():
        if isinstance(n,_quant_ops) or isinstance(n,DDPIntLinear) or isinstance(n,DDPLinearAllreduce):
            n.celibration()
            
def set_module_attr(net_proc, attr_dict:dict,name=''):
    keys = list(net_proc._modules.keys())
    if keys == []:
        if name in attr_dict.keys():
            set_attr(net_proc,attr_dict[name])
    else:
        for key in keys:
            set_module_attr(
                net_proc._modules[key],
                attr_dict,
                name=name + f'.{key}'
            )

def set_attr(net_proc,attr_dict:dict):
    for k,v in attr_dict.items():
        attrs = k.split('.') 
        current_module=net_proc
        for attr in attrs[:-1]:
            current_module = getattr(current_module, attr)
        if isinstance(v,(int,float)):
            v=torch.tensor(v,dtype=torch.float32).cuda()
        setattr(current_module,attrs[-1], v)

def split_celibration_ds(dataset,sampler='batch',batch=None):
    if isinstance(sampler,str) and sampler=='batch':
        ds=dataset.train_test_split(test_size=batch,shuffle=True)
        cel_ds=ds['test']
    elif isinstance(sampler,str) and sampler=='all':
        cel_ds = dataset
    elif isinstance(sampler,int):
        if sampler> len(dataset) or sampler<1:
            raise ValueError(f"sample nums must between 1 and {len(dataset)},but sampler nums is {sampler}")
        else:
            ds=dataset.train_test_split(test_size=sampler,shuffle=True)
            cel_ds=ds['test']
    else:
        raise ValueError(f"we only support batch、all or target nums sample func,but now is {sampler}")
    return cel_ds




    