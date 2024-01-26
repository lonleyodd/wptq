import torch 
import torch.nn as nn
import torch.nn.functional as F

class OIntInferSoftmax(nn.Module):
    def __init__(self, m):
        super(OIntInferSoftmax, self).__init__()
        self.G_value=None
        self.ops_type='softmax'

     
        self.const_n   = 15 
        self.const_bit = 16

        self.quantizer_x=m.quantizer_x

        self.scale_x=self.quantizer_x.scaler
        self.scale_y=torch.Tensor([1 / 2 ** (self.const_bit-1)]).detach().cuda()
        self.scale_last_layer=None

        self.input_shape=m.input_shape
        self.output_shape=m.output_shape
        self.div_bit=None

        self.int_infer_flag=False


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

    def div_shift(self,x,scale,lower,upper):
        div_shift_x=torch.round(x*scale)
        div_shift_x=torch.clamp(div_shift_x, lower,upper)
        return div_shift_x
      

    def _forward_infer(self, x_int):
        if self.div_bit is not None:
            x_int=self.div_shift(x_int,self.div_bit,self.quantizer_x.bit_lower_bound,self.quantizer_x.bit_upper_bound)
        
        scaling_factor=self.scale_x
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31-1)
        factor = torch.floor((2**31-1) / exp_int_sum)
        exp_int = torch.floor(exp_int * factor / 2 ** (31-self.const_bit+1))

        # exp_int=torch.clamp(exp_int,self.quantizer_y.bit_lower_bound,self.quantizer_y.bit_upper_bound)
        
        return exp_int
        
        
    def _forward_scale(self,scale_x=None):
        assert scale_x.size()==self.input_shape

        scale_last_layer=torch.unique(scale_x)
        
        if scale_last_layer!=self.scale_x:
            self.div_bit = torch.round(scale_last_layer/self.scale_x)

        scale_next_layer_tensor=torch.ones(self.output_shape,device=self.scale_y.device)*self.scale_y
        
        self.int_infer_flag=True
  
        return scale_next_layer_tensor