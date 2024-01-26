import torch
import torch.nn as nn

class BaseOps(nn.Module):
    def __init__(self,m) -> None:
        super().__init__()
        self.ops_name="BaseOps"
        self.int_infer_flag=False

        self.quantizer_w = m.quantizer_w
        self.quantizer_x = m.quantizer_x
        self.quantizer_b = m.quantizer_b
        self.quantizer_y = m.quantizer_y
        self.G_value     = m.G_value
        
        # for export onnx
        self.b_int = self.quantizer_b.quant(self.bias).detach() 
        self.w_int = self.quantizer_w.quant(self.weight).detach()
        
        self.scale_x=None
        self.scale_y=None
        self.scale_laster_layer=None
        
        # forward scale
        self.input_shape =None
        self.output_shape=None

        self.bit_lower_bound=None
        self.bit_upper_bound=None

    def forward(self,x):
        if self.int_infer_flag:
            x=self._forward_infer(x)
        else:
            x=self._forward_scale(x)
    
    def _forward_infer(self,x_int):
        
        x_int=self.div_shift(x_int,self.scale_laster_layer,self.bit_lower_bound,self.bit_upper_bound)

        y_int= x_int
        
        return y_int

    def _forward_scale(self,scale_x):
        assert scale_x.size()==self.input_shape
    
        scale_last_layer=scale_x.flatten()[0]

        scale_next_layer=torch.ones(self.output_shape)*self.scale_y
        
        self.int_infer_flag=True

        return scale_next_layer
    
    def div_shift(self,x,scale_y,scale_x):
        div_shift_x=torch.round(self.x*scale_y/scale_x)
        div_shift_x=torch.clamp(div_shift_x,self.bit_lower_bound,self.bit_upper_bound)
        return div_shift_x