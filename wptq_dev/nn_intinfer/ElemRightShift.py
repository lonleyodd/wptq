import torch
import torch.nn as nn

class OIntInferElemLeftShift(nn.Module):
    def __init__(self,m) -> None:
        super().__init__()
        self.ops_name = "ElemRightShift"
        self.int_infer_flag=False

        # shift_bit=2^n
        self.shift_bit = m.shift_bit
        self.quantizer_x = m.quantizer_x
        # self.quantizer_y = m.quantizer_y

        self.scale_x = self.quantizer_x.scaler
        # self.scale_y = self.quantizer_y.scaler
        self.scale_last_layer = None

        self.x_bit_lower_bound = self.quantizer_x.bit_lower_bound
        self.x_bit_upper_bound = self.quantizer_x.bit_upper_bound
        # self.y_bit_lower_bound = self.quantizer_y.bit_lower_bound
        # self.y_bit_upper_bound = self.quantizer_y.bit_upper_bound
        
        self.input_shape  = m.input_shape
        self.output_shape = m.output_shape


    def forward(self,x,const_n):
        if self.int_infer_flag:
            y=self._forward_infer(x)
        else:
            y=self._forward_scale(x)
        return y
    

    def _forward_infer(self,x_int):
        # if self.scale_last_layer is not None:
        #     int16_y = self.div_shift(x_int,self.scale_last_layer,self.scale_x,self.quantizer_x.bit_lower_bound,self.quantizer_x.bit_upper_bound)
        # else:
        #     int16_y = self.div_shift(x_int,1,self.quantizer_x.bit_lower_bound,self.quantizer_x.bit_upper_bound)

        # int8_y=torch.clamp(torch.round(int16_y*self.quantizer_x.scaler/self.scale_y),-128,127)
        int8_y = self.div_shift(x_int,self.scale_last_layer,self.scale_x,self.quantizer_x.bit_lower_bound,self.quantizer_x.bit_upper_bound)

        int8_y = torch.round(int8_y/self.shift_bit)
        
        return int8_y

    def _forward_scale(self,scale_x):
        assert scale_x.size()==self.input_shape
        
        scale_last_layer=torch.unique(scale_x)
        
        self.scale_last_layer=scale_last_layer
        
        # self.scale_x=self.scale_x*self.shift_bit

        scale_next_layer_tensor=torch.ones(self.output_shape,device=scale_x.device)*self.scale_x
        
        self.int_infer_flag=True
  
        return scale_next_layer_tensor
    

    def div_shift(self,x,scale1,scale2,lower,upper):
        if scale1!=scale2:
            div_shift_x = torch.round(x*scale1/self.scale_x)
            div_shift_x = torch.clamp(div_shift_x,lower,upper)
            return div_shift_x
        else:
            return x