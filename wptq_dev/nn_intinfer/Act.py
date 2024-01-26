import torch
import torch.nn as nn
from ..quant.quantizer import Quantizer

class OIntInferAct(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.ops_name="Act"
        self.int_infer_flag=False

        self.quant_type = m.quant_type
        self.quantizer_x = m.quantizer_x
        

        self.input_shape =m.input_shape
        self.output_shape=m.output_shape
        self.bit_lower_bound=self.quantizer_x.bit_lower_bound
        self.bit_upper_bound=self.quantizer_x.bit_upper_bound
        self.scale_x=self.quantizer_x.scaler
        self.scale_last_layer=None

    
    def forward(self,x):
        if self.int_infer_flag:
            y=self._forward_infer(x)
        else:
            y=self._forward_scale(x)
        return y

    def div_shift(self,x):
        div_shift_x=torch.round(x*self.scale_last_layer/self.scale_x)
        div_shift_x=torch.clamp(div_shift_x,self.bit_lower_bound,self.bit_upper_bound)
        return div_shift_x
    
    def _forward_infer(self,x):
        if self.quant_type=="quant_inputs":
            x=self.quantizer_x.quant(x)
        elif self.quant_type=="dequant_outputs":
            if self.scale_last_layer is not None:
                x=self.div_shift(x)
                x=self.quantizer_x.dequant(x)
        return x
    
    def _forward_scale(self,scale_x):
        if self.quant_type=="dequant_outputs":
            assert scale_x.size()==self.input_shape

            scale_last_layer=torch.unique(scale_x)
            
            self.scale_last_layer=scale_last_layer
        
        scale_next_layer_tensor=torch.ones(self.output_shape,device=scale_x.device)*self.scale_x

        self.int_infer_flag=True

        return scale_next_layer_tensor

    
 