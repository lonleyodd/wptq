import torch 
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizer import Quantizer

class QuantSoftmax(nn.Module):
    def __init__(self,
                ops_type='qnet',
                bit_type="int-8",
                
            ):
        super(QuantSoftmax, self).__init__()
        self.ops_name = 'softmax'
        self.ops_type = ops_type
        self.bit_type = "int-8"

        self.const_n = 15  
        self.const_bit = 16
        self.quantizer_x=Quantizer(self.ops_name,"input",bit_type=self.bit_type)
        self.quant=False

        self.input_shape=None
        self.output_shape=None
        
    def celibration(self):
        self.quantizer_x.celibration()
        self.G_value = torch.Tensor([1 / 2 ** (self.const_bit-1)]).cuda()
        self.quant=True

    def forward(self, x):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
            else:
                raise ValueError
        else:
            y=F.softmax(x,dim=-1) 
            self.quantizer_x.update_param(x)
            if self.input_shape is None or self.output_shape is None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y
        
    def _forward_qnet(self, x):
        x_int= self.quantizer_x.quant(x)
        x_fp = self.quantizer_x.dequant(x_int)
        y_fp = self._forward_onet(x) 
        
        y=F.softmax(x,dim=-1) 
        
        # info={'x':(x_int,x),'y':(y_fp,y)}
        # self.log_info(info)
        return y_fp
        
    def _forward_onet(self, x):
        x_int= self.quantizer_x.quant(x)

        scaling_factor=self.quantizer_x.scaler

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31-1)
        factor = torch.floor((2**31-1) / exp_int_sum)
        exp_int = torch.floor(exp_int * factor / 2 ** (31-self.const_bit+1))

        y=exp_int * self.G_value
        return y

    def int_exp_shift(self, x_int, scaling_factor):
        """
        x_fp=x_int*scale_factor
        q = torch.floor(x_int*scale_factor/scale_factor)
        r = x_int*scale_factor-q = x_int*scale-scaler*q=x_int*scale_factor-sacler*q
        2^rs = scaler_factor*r/2+1=scaler_factor[r/2+1/scaler_factor]=scaler_factor[r/2+x0_int]     
        """
        x_int = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 2 ** 4)

        x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.const_n * x0_int)

        q = torch.floor(x_int / x0_int)

        r = x_int - x0_int * q

        exp_int = r/2 - x0_int

        exp_int = torch.clamp(torch.floor(exp_int * 2 ** (self.const_n - q)), min=0)

        scaling_factor = scaling_factor / 2 ** self.const_n

        return exp_int, scaling_factor

    def _forward_intinfer(self,x):
        pass
    

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