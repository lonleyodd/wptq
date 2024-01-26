import torch.nn as nn
import torch
import torch.nn.functional as F
from ..quant.quantizer import Quantizer

class QuantGelu(nn.Module):
    def __init__(self,
                 ops_type='qnet',
                 bit_type='int-8',
                ):
        super(QuantGelu,self).__init__()
        self.ops_name="Gelu"
        self.ops_type=ops_type
        self.bit_type=bit_type
        self.quant=False

        # for onet
        self.const_k = 1.4142
        self.const_n = 15 
        self.const_output_bit=8
        self.const_sigmoid_factor=torch.Tensor([1 / 2 ** (self.const_output_bit-1)]).cuda()
        self.const_n=15
        self.quantizer_x = Quantizer(self.ops_name,"input",quant_bit=self.bit_type,quant_policy="per-channel")

        # for int infer
        self.input_shape=None
        self.output_shape=None

    def celibration(self):
        self.quantizer_x.celibration()
        self.quant=True
     
        
    def forward(self,x):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
        else:
            self.quantizer_x.update_param(x)
            y=F.gelu(x)
            if self.input_shape is None or self.output_shape is  None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y 

    
    def _forward_qnet(self,x):
        x_int = self.quantizer_x.quant(x)
        x_fp  = self.quantizer_x.dequant(x_int)

        y_fp  = self._forward_onet(x)

        y     = F.gelu(x)
        
        info  = {'x':(x_fp,x),'y':(y_fp,y)}

        self.log_info(info)

        return y

    def _forward_onet(self,x):
        scaling_factor=self.quantizer_x.scaler

        pre_x_int = self.quantizer_x.quant(x)

        scaling_factor_sig = scaling_factor * 1.702    

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max
        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)
        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max
        exp_int_sum.clamp_max_(2**31-1)
        factor = torch.floor((2 ** 31-1) / exp_int_sum)
        sigmoid_int = torch.floor(exp_int * factor / 2 ** (31-self.const_output_bit+1))
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.const_output_bit-1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        y=x_int*scaling_factor

        return y

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 2 ** 4)
        x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.const_n * x0_int)
        q = torch.floor(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (self.const_n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.const_n
        return exp_int, scaling_factor  
   

    def div_shift(self,x_int,scale,lower,upper):
        div_shift_x=torch.round(x_int*scale/self.scale_x)
        div_shift_x=torch.clamp(div_shift_x,lower,upper)
        return div_shift_x
    
    def log_info(self,info):
        self.logger.debug(f"module name: {self.module_name}, ops id: {self.ops_id}")
        self.logger.debug('------------------------------------------')
        for k,v in info.items():
            quant_x,x=v[0],v[1]
            loss=torch.nn.MSELoss()(quant_x,x)
            self.logger.debug(f"{self.ops_name} quant-dequant {k} loss: {loss.item()}")
        self.logger.debug('------------------------------------------\n')