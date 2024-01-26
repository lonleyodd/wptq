import torch.nn as nn
from ..quant.quantizer import Quantizer
import  torch.nn.functional  as F
import torch

class OIntInferLinear(nn.Linear):
    def __init__(self,m):
        super(OIntInferLinear, self).__init__(
            in_features=m.in_features,
            out_features=m.out_features, 
        )
        self.ops_name = "Linear"
        self.int_infer_flag=False
        
        self.quantizer_w = m.quantizer_w
        self.quantizer_x = m.quantizer_x
        self.quantizer_b = m.quantizer_b
        self.quantizer_y = m.quantizer_y
        self.G_value     = m.G_value
        self.bias        = m.bias
        self.weight      = m.weight

        self.b_int = self.quantizer_b.quant(self.bias).detach().cuda()
        self.w_int = self.quantizer_w.quant(self.weight).detach().cuda()

        self.scale_x = self.quantizer_x.scaler
        self.scale_b = self.quantizer_b.scaler
        self.scale_w = self.quantizer_w.scaler
        self.scale_y = self.quantizer_y.scaler
        self.scale_last_layer=None

        self.smooth_factor=self.quantizer_x.smooth_factor if hasattr(self.quantizer_x,"smooth_factor")  else None

        self.input_shape  = m.input_shape
        self.output_shape = m.output_shape
        
        self.x_bit_lower_bound=self.quantizer_x.bit_lower_bound
        self.x_bit_upper_bound=self.quantizer_x.bit_upper_bound
        self.y_bit_lower_bound=self.quantizer_y.bit_lower_bound
        self.y_bit_upper_bound=self.quantizer_y.bit_upper_bound

        
    def forward(self,x):
        if self.int_infer_flag:
            y=self._forward_infer(x)
        else:
            y=self._forward_scale(x)
        return y

    
    def _forward_infer(self,x_int):
        
        x_int = self.div_shift(x_int,self.scale_last_layer,self.scale_x,self.x_bit_lower_bound,self.x_bit_upper_bound)

        int_y = F.linear(x_int,self.w_int,self.b_int)
        
        assert self.G_value is not None, "please check your G_value"
        
        int_y = (int_y/self.G_value).round().clip(min=self.y_bit_lower_bound, 
                    max=self.y_bit_upper_bound)
        return int_y

    def _forward_scale(self,scale_x):
        assert scale_x.size()==self.input_shape

        device=scale_x.device
        
        scale_last_layer=torch.unique(scale_x)

        if self.smooth_factor is not None:
            self.scale_x=self.scale_x*self.smooth_factor

        self.scale_last_layer=scale_last_layer
    
        scale_next_layer=torch.ones(self.output_shape,device=device)*self.scale_y
        
        self.int_infer_flag=True

        return scale_next_layer

    def div_shift(self,x,scale1,scale2,lower,upper):
        # 尽可能使用2^n
        if scale1.shape!= scale2.shape or scale1!= scale2:
            div_shift_x=torch.round(x*scale1/scale2)
            div_shift_x=torch.clamp(div_shift_x, lower,upper)
            return div_shift_x
        else:
            return x