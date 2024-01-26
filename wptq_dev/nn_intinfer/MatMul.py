import torch.nn as nn
import torch

class OIntInferMatMul(nn.Module):
    def __init__(self,m) -> None:
        super().__init__()
        self.ops_name = "MatMul"
        self.int_infer_flag=False
        
        self.quantizer_x1 = m.quantizer_x1
        self.quantizer_x2 = m.quantizer_x2
        self.quantizer_y  = m.quantizer_y
        self.G_value      = m.G_value

        self.scale1_last_layer = None
        self.scale2_last_layer = None    
        
        self.scale_x1 = self.quantizer_x1.scaler
        self.scale_x2 = self.quantizer_x2.scaler
        self.scale_y  = self.quantizer_y.scaler
        
        self.input1_shape=m.input1_shape
        self.input2_shape=m.input2_shape
        self.output_shape=m.output_shape

        self.x1_bit_lower_bound = self.quantizer_x1.bit_lower_bound
        self.x1_bit_upper_bound = self.quantizer_x1.bit_upper_bound
        self.x2_bit_lower_bound = self.quantizer_x2.bit_lower_bound
        self.x2_bit_upper_bound = self.quantizer_x2.bit_upper_bound
        

    def forward(self,x1,x2):
        if self.int_infer_flag:
            y=self._forward_infer(x1,x2)
        else:
            y=self._forward_scale(x1,x2)
        return y
    

    def _forward_infer(self,x1_int,x2_int):
        x1_int=self.div_shift(x1_int,self.scale1_last_layer,self.scale_x1,self.x1_bit_lower_bound,self.x1_bit_upper_bound)
    
        x2_int=self.div_shift(x2_int,self.scale2_last_layer,self.scale_x2,self.x2_bit_lower_bound,self.x2_bit_upper_bound)

        y_int=(x1_int@x2_int/self.G_value).round().clip(min=self.quantizer_y.bit_lower_bound,
                            max=self.quantizer_y.bit_upper_bound)
        return y_int
   

    def _forward_scale(self,scale_x1,scale_x2):
        assert scale_x1.size()==self.input1_shape
        assert scale_x2.size()==self.input2_shape
        
        # scale1_last_layer=scale_x1.flatten()[0]
        # scale2_last_layer=scale_x2.flatten()[0]

        scale1_last_layer=torch.unique(scale_x1)
        scale2_last_layer=torch.unique(scale_x2)

        self.scale1_last_layer=scale1_last_layer
        self.scale2_last_layer=scale2_last_layer
      
        scale_next_layer=torch.ones(self.output_shape,device=self.scale_y.device)*self.scale_y
 
        self.int_infer_flag=True
  
        return scale_next_layer
    

    def div_shift(self,x,scale1,scale2,lower,upper):
        if scale1!=scale2:
            div_shift_x=torch.round(x*scale1/scale2)
            div_shift_x=torch.clamp(div_shift_x, lower,upper)
            return div_shift_x
        else:
            return x

        

