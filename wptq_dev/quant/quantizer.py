import torch
from torch import Tensor

support_ops_list=[
    'linear', 'conv2d', 'layernorm','gelu','softmax',
    'matmul', 'elemadd','act','elemrightshift'
]

support_quant_policy_list=[
    "per-layer",
    "per-channel",
    "per-group"
]

def get_bite_bound(bit_name):
    bit_type,bit_width=bit_name.split('-')
    if bit_type!="int" and bit_type!="uint":
        raise ValueError(f"Invalid quant bit type: {bit_name}")
    
    bit_width=int(bit_width)

    upper_bound=2**(bit_width-1)-1 if bit_type=="int" else  2**bit_width-1 
    lower_bound=-2**(bit_width-1) if bit_type=="int" else 0  
    return upper_bound,lower_bound

def reshape_tensor(x,ops):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if ops=="per-channel":
        x=x.detach().reshape(-1,x.size(-1))
    elif ops=="per-layer":
        x=x.detach().reshape(-1,x.size(-1))
    elif ops=="per-token":
        x=x.detach().reshape(x.shape[1],-1).transpose(0,1)
    return x

def find_max_min(x,smooth,threshold=None):
    if smooth=="do_perc_clip": 
        assert threshold is not None
        bins_length=10000
        hist=torch.histc(x,bins=bins_length)
        edges = torch.linspace(x.min(), x.max(), steps=len(hist)+1)
        cumsum= torch.cumsum(hist, dim=0)
        min_count = cumsum[-1]*(1-threshold)
        T=torch.nonzero(cumsum <= min_count)
        min_index = T[-1] if T.size()[0]>0 else T[0]
        min_value = edges[min_index+1].cuda()
        max_count = cumsum[-1]*threshold
        max_index = torch.nonzero(cumsum >= max_count)[0]
        max_value = edges[max_index+1].cuda()
    elif smooth=="do_value_clip":
        assert threshold is not None
        max_value = torch.tensor(threshold,dtype=torch.float32,device='cuda') if x.max()>threshold else x.max()
        min_value = torch.tensor(-threshold,dtype=torch.float32,device='cuda') if x.min()<-threshold else x.min()  
    else:
        raise ValueError(f"invalid smooth func {smooth}")
    return min_value,max_value

class QuantBaseCfg():
    def __init__(self,*args,**kwargs) -> None:
        self.quant= kwargs.get("quant",True)
        self.quant_bit=kwargs.get("quant_bit","int-8")
        self.quant_policy=kwargs.get("quant_policy","per-layer")
        self.quant_strategy=kwargs.get("quant_strategy",None)
        
        self.smooth=kwargs.get("smooth",None)
        self.quant_threshold=kwargs.get("threshold",None)
        self.symmetric=kwargs.get("symmetric",True)
        self.eps=1e-5

        # self.update(**kwargs)
      
    def update(self,**kwargs):
        for k,v in kwargs.items():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                raise ValueError(f"Invalid key {k} in QuantBaseCfg")
            

class Quantizer():
    def __init__(self,
                 ops:str,
                 obj:str="weight",
                 **kwargs,
        ) -> None:
        
        assert ops.lower() in support_ops_list,f"ops {ops} unspport quantization, "
        self.ops=ops

        assert obj.lower() in ["weight","bias","output","input"],"obj is error "
        self.obj=obj

        self.quant_cfg = QuantBaseCfg(**kwargs)
        self.bit_upper_bound,self.bit_lower_bound=get_bite_bound(self.quant_cfg.quant_bit)
        
        self.scaler     = None
        self.zero_point = None
        self.min_val    = None
        self.max_val    = None
        
    def celibration(self,):
        self.scaler,self.zero_point=self.get_quantization_params()
        if self.quant_cfg.quant_policy=="per-token":
            self.scaler=self.scaler.detach().view(-1,1)
            self.zero_point=self.zero_point.detach().view(-1,1)
            

    def update_param(self,x:Tensor):
        if self.quant_cfg.quant_strategy!='smooth':
            if self.obj==("weight" or "bias") and \
                self.max_val is not None and \
                self.min_val is not None:
                return 
        
            x = reshape_tensor(x,self.quant_cfg.quant_policy)
            cur_max=x.max(dim=0)[0]

            if self.max_val is None:
                self.max_val = cur_max
            else:
                self.max_val = torch.max(cur_max, self.max_val)

            cur_min = x.min(dim=0)[0]
            
            if self.min_val is None:
                self.min_val = cur_min
            else:
                self.min_val = torch.min(cur_min, self.min_val)

            if self.quant_cfg.quant_policy=="per-layer":
                if self.quant_cfg.smooth is not None:
                    batch_min_val,batch_max_val=find_max_min(x,
                                                            smooth=self.quant_cfg.smooth,
                                                            threshold=self.quant_cfg.quant_threshold)
                    self.max_val=min(batch_max_val,self.max_val.min())
                    self.min_val=max(batch_min_val,self.min_val.max())
                else:
                    self.max_val=self.max_val.max()
                    self.min_val=self.min_val.min()
        else:
            x = reshape_tensor(x,self.quant_cfg.quant_policy)
            if self.obj==("weight" or "bias") and \
                self.max_val is not None and \
                self.min_val is not None:
                return   
            if self.obj=="weight":
                self.max_val=x.abs().max(dim=0,keepdim=True)[0].float().cpu()
                self.min_val=x.abs().min(dim=0,keepdim=True)[0].float().cpu()
            elif self.obj=="output":
                x= reshape_tensor(x,self.policy)
                max_val=x.abs().max(dim=0)[0].float().cpu()
                if self.max_val is None:
                    self.max_val=max_val
                else:
                    self.max_val=torch.max(self.max_val,max_val)
                min_val=x.abs().min(dim=0)[0].float().cpu()
                if self.min_val is None:
                    self.min_val=min_val
                else:
                    self.min_val=torch.min(self.min_val,min_val)
            else:
                max_val=x.abs().max(dim=0)[0].float().cpu()
                
                if not hasattr(self,'act_scales'):
                    setattr(self,'act_scales',max_val)
                else:
                    self.act_scales=torch.max(self.act_scales,max_val)
            
                batch_max_val = x.max(dim=0)[0].float().cpu()

                if self.max_val is None:
                    self.max_val=batch_max_val
                else:
                    self.max_val=torch.max(self.max_val,batch_max_val)  

                batch_min_val = x.min(dim=0)[0].float().cpu()
                if self.min_val is None:
                    self.min_val=batch_min_val
                else:
                    self.min_val=torch.min(self.min_val,batch_min_val) 
                
            
    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_upper_bound
        qmin = self.bit_lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.quant_cfg.symmetric:
            max_val = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_val / round(((qmax - qmin) / 2))
            scale.clamp_(self.quant_cfg.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.quant_cfg.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
        

    def quant(self,x,scale=None,zero_point=None):
        if not self.quant_cfg.quant:
            return x
        device,dtype=x.device,x.dtype
        scale = self.scaler.to(x.device) if scale is None else scale
        if zero_point is None:
            zero_point = self.zero_point.to(x.device)
        outputs = x / scale + zero_point
        outputs = outputs.round().clamp(self.bit_lower_bound, self.bit_upper_bound).to(device,dtype)
        return outputs

    def dequant(self,x,scale=None,zero_point=None):
        if not self.quant_cfg.quant:
            return x
        device,dtype=x.device,x.dtype
        if scale is None:
            scale = self.scaler.to(x.device)
        if zero_point is None:
            zero_point = self.zero_point.to(device)
        outputs=(x - zero_point) * scale
        outputs=outputs.to(device,dtype)
        return outputs
    