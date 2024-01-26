import torch.nn as nn
import torch
class OIntInferLayerNorm(nn.LayerNorm):
    def __init__(self, m):
        super().__init__ (
            normalized_shape=m.normalized_shape,
            eps=m.eps,
            elementwise_affine=m.elementwise_affine
        )
        self.ops_type = "LayerNorm"
        self.int_infer_flag=False

        if self.elementwise_affine:
            self.weight=m.weight
            self.bias=m.bias
        else:
            self.register_parameter("weight",None)
            self.register_parameter("bias",None)

        self.quantizer_x = m.quantizer_x


        self.scale_x    = m.quantizer_x.scaler
        self.scale_y    = m.scale_y
        self.scale_last_layer=None

        self.w_int = torch.round(self.weight*self.scale_y).detach()
        self.b_int = torch.round(self.bias*self.scale_y).detach()

        self.input_shape=m.input_shape
        self.output_shape=m.output_shape

        self.const_dim_sqrt=torch.round(torch.sqrt(torch.tensor(768,dtype=torch.float)).detach().cuda())
        self.const_scale_factor=self.const_dim_sqrt/2**20

        self.x_bit_lower_bound=m.quantizer_x.bit_lower_bound
        self.x_bit_upper_bound=m.quantizer_x.bit_upper_bound


    def forward(self,x):
        if self.int_infer_flag:
            y=self._forward_infer(x)
        else:
            y=self._forward_scale(x)
        return y
    

    def _forward_infer(self,x_int):
        x_int= self.div_shift(x_int,self.scale_last_layer,self.scale_x,self.x_bit_lower_bound,self.x_bit_upper_bound)
        
        x_int_mean= torch.round(x_int.mean(axis=2,keepdim=True))
        x_sub = x_int-x_int_mean
        y_sq_int =x_sub**2
        I_var=torch.sum(y_sq_int, axis=2, keepdim=True)

        k_i = 2**8
        for i in range(10):
            k_i=torch.floor((k_i+torch.floor(I_var/k_i))/2)
        std_int = k_i  

        y_int = torch.round(x_sub*self.w_int*self.const_dim_sqrt)
        y_int = torch.round(y_int/std_int)
        y_int = torch.round(y_int+self.b_int)

        return y_int
    

    def _forward_scale(self,scale_x): 
        assert scale_x.size()==self.input_shape
    
        scale_last_layer=scale_x.flatten()[0]
        
        self.scale_last_layer=scale_last_layer

        scale_next_layer_tensor=torch.ones(self.output_shape,device=scale_x.device)*(1/self.scale_y)
        
        self.int_infer_flag=True
  
        return scale_next_layer_tensor


    def div_shift(self,x_int,scale1,scale2,lower,upper):
        if scale1!=scale2:
            div_shift_x=torch.round(x_int*scale1/scale2)
            div_shift_x=torch.clamp(div_shift_x,lower,upper)
            return div_shift_x
        else:
            return x_int