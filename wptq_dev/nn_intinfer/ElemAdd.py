import torch
import torch.nn as nn
from ..quant.quantizer import Quantizer

class OIntInferElemAdd(nn.Module):
    def __init__(self,m):
        super().__init__()
       
        self.quant=False
        self.quant_x_flag=False
        self.int_infer_flag=False

        self.quantizer_y  = m.quantizer_y
        self.quantizer_x1 = m.quantizer_x1
        self.quantizer_x2 = m.quantizer_x2
        
        self.input1_shape=m.input1_shape
        self.input2_shape=m.input2_shape
        self.output_shape=m.output_shape

        self.scale_x1=self.quantizer_x1.scaler
        self.scale_x2=self.quantizer_x2.scaler
        self.scale_y=self.quantizer_y.scaler
        self.scale_last_layer1=None
        self.scale_last_layer2=None
    
    def forward(self,x1,x2):
        if self.int_infer_flag:
            y=self._forward_infer(x1,x2)
        else:
            y=self._forward_scale(x1,x2)
        return y

    def div_shift(self,x,scale1,scale2,lower,upper):
        if scale1.shape!=scale2.shape or scale1!=scale2:
            div_shift_x=torch.round(x*scale1/scale2)
            div_shift_x=torch.clamp(div_shift_x, lower,upper)
            return div_shift_x
        else:
            return x

    def _forward_infer(self,x1_int,x2_int):
        
        x1_int=self.div_shift(x1_int,self.scale_last_layer1,self.scale_x1,self.quantizer_x1.bit_lower_bound,self.quantizer_x1.bit_upper_bound)
        x2_int=self.div_shift(x2_int,self.scale_last_layer2,self.scale_x2,self.quantizer_x2.bit_lower_bound,self.quantizer_x2.bit_upper_bound)

        scale_x1=self.scale_x1/self.scale_y
        scale_x2=self.scale_x2/self.scale_y

        y_int=torch.round(x1_int*scale_x1)+torch.round(x2_int*scale_x2)

        # y_int=torch.clamp(y_int,self.quantizer_y.bit_lower_bound,self.quantizer_y.bit_upper_bound)
        
        return y_int
    
    def _forward_scale(self,scale_x1,scale_x2):
        assert scale_x1.size()==self.input1_shape
        assert scale_x2.size()==self.input2_shape

        if torch.unique(scale_x1).shape[0]!=1:
            scale_last_layer1=scale_x1[0,0]
        else:
            scale_last_layer1=torch.unique(scale_x1)

        scale_last_layer2=torch.unique(scale_x2)
       
        self.scale_last_layer1=scale_last_layer1
      
        self.scale_last_layer2=scale_last_layer2

        scale_next_layer_tensor=torch.ones(self.output_shape,device=scale_x1.device)*self.scale_y

        self.int_infer_flag=True

        return scale_next_layer_tensor

    
 