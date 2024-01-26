import torch
import torch.nn as nn
from ..quant.quantizer import Quantizer

class QuantElemRightShift(nn.Module):
    def __init__(self,
                 ops_type='qnet',
                 quant_bit='int-8'
                 ) -> None:
        super().__init__()
        self.ops_name="ElemRightShift"
        self.ops_type=ops_type
        self.quant_bit=quant_bit

        self.quant=False
        self.quantizer_x=Quantizer(self.ops_name,"input",quant_bit="int-8")
      

        self.input_shape =None
        self.output_shape=None    

    def celibration(self):
        self.quantizer_x.celibration()
        # self.quantizer_y.celibration()
        self.quant=True

    def forward(self,x,shift_bit):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x,shift_bit)
            elif self.ops_type =='onet':
                return self._forward_onet(x,shift_bit)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x,shift_bit)
        else:
            setattr(self,"shift_bit",shift_bit)
            y=x/shift_bit
            self.quantizer_x.update_param(x)
            # self.quantizer_y.update_param(y)

            if self.input_shape is None or self.output_shape is  None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y 

    def _forward_qnet(self,x,shift_bit):
        x_int8 = self.quantizer_x.quant(x)

        x_int8 = torch.round(x_int8/shift_bit)
        
        x_fp =self.quantizer_x.dequant(x_int8)
        
        return x_fp
    
    def _forward_onet(self,x,shift_bit):
        
        x_int8 = self.quantizer_x.quant(x)

        x_int8 = torch.round(x_int8/shift_bit)
        
        x_fp =self.quantizer_x.dequant(x_int8)

        return x_fp
    
    def _forward_intinfer(self,x,shift_bit):
        pass