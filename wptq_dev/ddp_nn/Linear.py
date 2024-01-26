import torch.nn as nn
import torch
import  torch.nn.functional as F
from torch import Tensor
from ..quant.quantizer import Quantizer
from deepspeed.module_inject.layers import LinearLayer,LinearAllreduce
from deepspeed import comm as dist
import os
class DDPIntLinear(nn.Module):
    def __init__(self,
                 ops:LinearLayer,
                 ops_type="int-8",
                 **kwargs):
        super(DDPIntLinear, self).__init__()
        self.weight=ops.weight
        if ops.bias is not None:
            self.bias = ops.bias
        else:
            self.bias = nn.Parameter(torch.zeros(self.weight.size(0),dtype=self.weight.dtype, device=self.weight.device))

        self.ops_name ="Linear"
        self.ops_type =ops_type

        self.quant = False
        
        self.quantizer_x=Quantizer(self.ops_name,"input",**kwargs["input"])
        self.quantizer_y=Quantizer(self.ops_name,"output",**kwargs["output"])
        self.quantizer_b=Quantizer(self.ops_name,"bias",**kwargs["bias"])
        self.quantizer_w=Quantizer(self.ops_name,"weight",**kwargs["weight"])

        # for onet ops
        self.input_shape=None
        self.output_shape=None
        self.G_value = None


    def celibration(self,):
        self.quantizer_x.celibration()
        self.quantizer_w.celibration()
        self.quantizer_b.celibration()
        self.quantizer_y.celibration()

        if hasattr(self.quantizer_x,'smooth_factor'):
            factor= torch.round(self.quantizer_x.smooth_factor)
            self.quantizer_x.scaler=(self.quantizer_x.scaler.to(factor.device)/factor).max()
            self.quantizer_w.scaler=(self.quantizer_w.scaler.to(factor.device)*factor).max()

        if self.ops_type =='onet':
            self.G_value = torch.round(self.quantizer_y.scaler.to(self.quantizer_x.scaler.device)/(self.quantizer_x.scaler*self.quantizer_w.scaler))
            self.quantizer_b.scaler= self.quantizer_x.scaler*self.quantizer_w.scaler
        self.quant=True

    def forward(self, x:Tensor):
        if self.quant:
            if self.ops_type == 'qnet':
                return self._forward_qnet(x)
            elif self.ops_type == 'onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
        else:
            output = torch.matmul(x, self.weight.transpose(-1, -2))
            if self.bias is not None:
                output += self.bias

            self.quantizer_x.update_param(x)
            self.quantizer_w.update_param(self.weight)
            self.quantizer_b.update_param(self.bias)
            self.quantizer_y.update_param(output)
            return output  

    def _forward_qnet(self,x):
        if hasattr(self.quantizer_x,'smooth_factor'): 
            x=x/self.quantizer_x.smooth_factor
        
        int_x = self.quantizer_x.quant(x)
        fp_x  = self.quantizer_x.dequant(int_x)

        int_weight = self.quantizer_w.quant(self.weight)
        fp_weight  = self.quantizer_w.dequant(int_weight)

        int_bias   = self.quantizer_b.quant(self.bias)
        fp_bias    = self.quantizer_b.dequant(int_bias)

        y = torch.matmul(fp_x, fp_weight.transpose(-1, -2))

        if self.bias is not None:
            y += fp_bias
        
        int_y=self.quantizer_y.quant(y)
        fp_y=self.quantizer_y.dequant(int_y)

        if os.environ["RANK"]=='0':
            info={'x':(x,fp_x),'weight':(self.weight,fp_weight),'bias':(self.bias,fp_bias),'y':(y,fp_y)}
            self.log_info(info)
        return fp_y
    
    def log_info(self,info):
        self.logger.debug(f"module name: {self.module_name}, ops id: {self.ops_id}")
        self.logger.debug('------------------------------------------')
        for k,v in info.items():
            if k=="G_value":
                self.logger.debug(f"{self.ops_name} {k} : {v.item()}")
            else:
                quant_x,x=v[0],v[1]
                loss=torch.nn.MSELoss()(quant_x,x)
                self.logger.debug(f"{self.ops_name} quant-dequant {k} loss: {loss.item()}")
        self.logger.debug('------------------------------------------\n')

class DDPLinearAllreduce(nn.Module):
    def __init__(self,
                 ops:LinearAllreduce,
                 ops_type="qnet",
                 **kwargs):
        super(DDPLinearAllreduce, self).__init__()
        self.weight=ops.weight
        if ops.bias is not None:
            self.bias = ops.bias
        else:
            self.bias = nn.Parameter(torch.zeros(self.weight.size(0),dtype=self.weight.dtype, device=self.weight.device))
        self.mp_group = ops.mp_group
        self.ops_name ="Linear"
        self.ops_type =ops_type
        
        self.quant = False
        
        self.quantizer_x=Quantizer(self.ops_name,"input",**kwargs["input"])
        self.quantizer_y=Quantizer(self.ops_name,"output",**kwargs["output"])
        self.quantizer_b=Quantizer(self.ops_name,"bias",**kwargs["bias"])
        self.quantizer_w=Quantizer(self.ops_name,"weight",**kwargs["weight"])

        # for onet ops
        self.input_shape=None
        self.output_shape=None
        self.G_value = None

    def celibration(self,):
        self.quantizer_x.celibration()
        self.quantizer_w.celibration()
        self.quantizer_b.celibration()
        self.quantizer_y.celibration()

        if hasattr(self.quantizer_x,'smooth_factor'):
            factor= self.quantizer_x.smooth_factor
            self.quantizer_x.scaler=(self.quantizer_x.scaler.to(factor.device)/factor).max()
            # self.quantizer_w.scaler=(self.quantizer_w.scaler.to(factor.device)*factor).max()
            self.quantizer_w.scaler=self.weight.abs().clip(1e-8,None).max(dim=1)[0]

        if self.ops_type =='onet':
            self.G_value = torch.round(self.quantizer_y.scaler.to(self.quantizer_x.scaler.device)/(self.quantizer_x.scaler*self.quantizer_w.scaler))
            self.quantizer_b.scaler= self.quantizer_x.scaler*self.quantizer_w.scaler

        self.quant=True

    def _forward_smooth(self,x):
        if hasattr(self.quantizer_x,'smooth_factor'):
            factor=self.quantizer_x.smooth_factor
            x_div=x/factor
        else:
            x_div=x

        x_scales = x_div.abs().max()
        if self.mp_group is not None:
            x_scales_list=[torch.zeros(1,dtype=torch.bfloat16).to(x_scales.device) for i in range(torch.distributed.get_world_size())]
            dist.all_gather(x_scales_list, x_scales)
            x_scales=torch.cat(x_scales_list).max()
            
        q_max = 127
        x_scales.clamp_(min=1e-5).div_(q_max)
        fp_x=x_scales*torch.round((x_div/x_scales)).to(self.weight.dtype)

        w_scales = self.weight.abs().max()
        if self.mp_group is not None:
            w_scales_list=[torch.zeros(1,dtype=torch.bfloat16).to(w_scales.device) for i in range(torch.distributed.get_world_size())]
            dist.all_gather(w_scales_list, w_scales)
            w_scales=torch.cat(w_scales_list).max()
        
        q_max = 127
        w_scales.clamp_(min=1e-5).div_(q_max)
        fp_weight=w_scales*torch.round(self.weight/w_scales)
    
        
        fp_y = torch.matmul(fp_x, fp_weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(fp_y, group=self.mp_group)
        if self.bias is not None:
            fp_y += self.bias
        
        return fp_y
     
    def forward(self, x:Tensor):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)    
        else:
            output = torch.matmul(x, self.weight.transpose(-1, -2))
            if self.mp_group is not None:
                dist.inference_all_reduce(output, group=self.mp_group)
            if self.bias is not None:
                output += self.bias
            self.quantizer_x.update_param(x)
            self.quantizer_w.update_param(self.weight)
            self.quantizer_b.update_param(self.bias)
            self.quantizer_y.update_param(output)
            return output   
          
    def _forward_qnet(self,x):
        if hasattr(self.quantizer_x,'smooth_factor'): 
            x=x/self.quantizer_x.smooth_factor

        int_weight = self.quantizer_w.quant(self.weight)
        fp_weight  = self.quantizer_w.dequant(int_weight)

        int_bias   = self.quantizer_b.quant(self.bias)
        fp_bias    = self.quantizer_b.dequant(int_bias)

        int_x      = self.quantizer_x.quant(x)
        fp_x       = self.quantizer_x.dequant(int_x)

        y = torch.matmul(fp_x, fp_weight.transpose(-1, -2))
        
        int_y=self.quantizer_y.quant(y)
        fp_y=self.quantizer_y.dequant(int_y)

        if self.mp_group is not None:
            dist.inference_all_reduce(fp_y, group=self.mp_group)
        if self.bias is not None:
            fp_y += fp_bias

        # if os.environ["RANK"]=='0':
        #     info={'x':(x,fp_x),'weight':(self.weight,fp_weight),'bias':(self.bias,fp_bias),'y':(y,fp_y)}
        #     self.log_info(info)
        return fp_y

    def log_info(self,info):
        self.logger.debug(f"module name: {self.module_name}, ops id: {self.ops_id}")
        self.logger.debug('------------------------------------------')
        for k,v in info.items():
            if k=="G_value":
                self.logger.debug(f"{self.ops_name} {k} : {v.item()}")
            else:
                quant_x,x=v[0],v[1]
                loss=torch.nn.MSELoss()(quant_x,x)
                self.logger.debug(f"{self.ops_name} quant-dequant {k} loss: {loss.item()}")
        self.logger.debug('------------------------------------------\n')