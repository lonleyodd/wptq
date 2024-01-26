import torch
import torch.nn as nn
from ..quant.quantizer import Quantizer

class QuantAct(nn.Module):
    def __init__(self,
                 quant_type:str,
                 ops_type='qnet',
                 quant_bit='int-8'
                 ):
        super().__init__()
        self.ops_name="Act"
        self.quant_type=quant_type
        self.ops_type=ops_type
        self.quant_bit=quant_bit

        self.quant=False
        
        self.quantizer_x = Quantizer(self.ops_name,"input",quant_bit=self.quant_bit)

        self.input_shape=None
        self.output_shape=None

    def celibration(self):
        self.quant=True
        self.quantizer_x.celibration()
   
        
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
            if self.input_shape is None or self.output_shape is None:
                self.input_shape=x.size()
                self.output_shape=x.size()
            return x

    def _forward_qnet(self,x):
        return x
    
    def _forward_onet(self,x):
        int_x=self.quantizer_x.quant(x)

        fp_x=self.quantizer_x.dequant(int_x)

        return fp_x
    
    def _forward_intinfer(self,x):
        pass