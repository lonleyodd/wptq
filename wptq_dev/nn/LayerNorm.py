import torch.nn as nn
import torch
import torch.nn.functional as F
from ..quant.quantizer import Quantizer

class QuantLayerNorm(nn.LayerNorm):
    def __init__(self, 
                 ops: nn.LayerNorm,
                 ops_type="qnet",
                 bit_type="int-16",
                 hidden_dim=768
                 ):
        super().__init__ (
            normalized_shape=ops.normalized_shape,
            eps=ops.eps,
            elementwise_affine=ops.elementwise_affine
        )
        if self.elementwise_affine:
            self.weight=ops.weight
            self.bias=ops.bias
        else:
            self.register_parameter("weight",None)
            self.register_parameter("bias",None)
        self.ops_name="LayerNorm"
        self.bit_type=bit_type
        self.ops_type=ops_type


        self.quant=False
        self.scale_y=127
        self.quantizer_x=Quantizer(self.ops_name,"input",quant_bit=self.bit_type)

        self.const_dim_sqrt=torch.round(torch.sqrt(torch.tensor(hidden_dim,dtype=torch.float)).cuda())
        self.input_shape=None
        self.output_shape=None

    def celibration(self):
        self.quant=True
        self.quantizer_x.celibration()
    

    def forward(self,x):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_onet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
        else:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,self.eps)
            self.quantizer_x.update_param(x)
            if self.input_shape is None or self.output_shape is  None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y 
    
    def _forward_qnet(self,x):
        x_int=self.quantizer_x.quant(x)
        
        x_fp =self.quantizer_x.dequant(x_int)

        y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,self.eps)
        y_fp = self._forward_onet(x)

        # info ={'x':(x_fp,x),'y':(y,y_fp)}
        # self.log_info(info)
        return y_fp


    def _forward_onet(self,x):
        x_int=self.quantizer_x.quant(x)

        b_int=torch.round(self.bias.data.detach()*self.scale_y)
        w_int=torch.round(self.weight.data.detach()*self.scale_y)
        
        x_int_mean= torch.round(x_int.mean(axis=2,keepdim=True))
        x_sub = x_int-x_int_mean
        y_sq_int =x_sub**2
        I_var=torch.sum(y_sq_int, axis=2, keepdim=True)

        k_i = 2**8
        for i in range(10):
            k_i=torch.floor((k_i+torch.floor(I_var/k_i))/2)
        std_int = k_i  

        y_int = torch.round(x_sub*w_int*self.const_dim_sqrt)
        y_int = torch.round(y_int/std_int)
        y_int = torch.round(y_int+b_int)

        y_fp = y_int/self.scale_y

        return y_fp
    
    def _forward_intinfer(self,x):
        pass

    def log_info(self,*args,**kwargs):
        self.logger.debug(f"module name: {self.module_name}, ops id: {self.ops_id}")
        self.logger.debug('------------------------------------------')
        for k,v in kwargs.items():
            quant_x,x=v[0],v[1]
            loss=torch.nn.MSELoss()(quant_x,x)
            self.logger.debug(f"{self.ops_name} quant-dequant {k} loss: {loss.item()}")
        self.logger.debug('------------------------------------------\n')