import torch.nn as nn
import torch

class OIntInferGelu(nn.Module):
    def __init__(self,m):
        super(OIntInferGelu,self).__init__()
        self.ops_name="GELU"
        self.int_infer_flag=False
        self.const_k = 1.4142
        self.const_n = 15 
        self.const_output_bit=8
        self.const_sigmoid_factor=torch.Tensor([1 / 2 ** (self.const_output_bit-1)]).cuda()
        self.const_n=15



        self.quantizer_x = m.quantizer_x
    
        self.scale_x = self.quantizer_x.scaler
        self.scale_y = self.scale_x* self.const_sigmoid_factor
        self.scale_last_layer=None

        self.input_shape  = m.input_shape
        self.output_shape = m.output_shape
        
        self.x_bit_lower_bound=self.quantizer_x.bit_lower_bound
        self.x_bit_upper_bound=self.quantizer_x.bit_upper_bound
        
        
    def forward(self,x):
        if self.int_infer_flag:
            y=self._forward_infer(x)
        else:
            y=self._forward_scale(x)
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
    

    def _forward_infer(self,x_int):
        pre_x_int = self.div_shift(x_int,self.scale_last_layer,self.scale_x,self.x_bit_lower_bound,self.x_bit_upper_bound)
        
        scaling_factor_sig = self.scale_x * 1.702    

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31-1)
        factor = torch.floor((2 ** 31-1) / exp_int_sum)
        sigmoid_int = torch.floor(exp_int * factor / 2 ** (31-self.const_output_bit+1))

        x_int = pre_x_int * sigmoid_int
        
        return x_int
            
    def _forward_scale(self,scale_x):
        assert scale_x.size()==self.input_shape
        
        scale_last_layer=scale_x.flatten()[0]
        
        # if scale_last_layer !=self.scale_x:
        self.scale_last_layer=scale_last_layer


        scale_next_layer_tensor=torch.ones(self.output_shape,device=self.scale_y.device)*self.scale_y
        
        self.int_infer_flag=True
  
        return scale_next_layer_tensor

    def div_shift(self,x_int,scale1,scale2,lower,upper):
        if scale1!=scale2:
            div_shift_x=torch.round(x_int*scale1/scale2)
            div_shift_x=torch.clamp(div_shift_x,lower,upper)
            return div_shift_x
        else:
            return x_int