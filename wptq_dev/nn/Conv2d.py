import torch
import torch.nn.functional as F
import torch.nn as nn
from ..quant.quantizer import Quantizer

class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 ops:nn.Conv2d,
                 ops_type="qnet",
                 bit_type="int-8"
                ):
        super(QuantConv2d, self).__init__(
            in_channels=ops.in_channels,
            out_channels=ops.out_channels,
            kernel_size=ops.kernel_size,
            stride=ops.stride,
            padding=ops.padding,
            dilation=ops.dilation,
            groups=ops.groups, 
        )

        self.ops_name='Conv2d'
        self.ops_type=ops_type
        self.bit_type=bit_type

        self.weight = ops.weight
        self.bias   = ops.bias
        
        self.quant = False
        self.quantizer_w=Quantizer("Conv2d","weight",bit_type=self.bit_type)
        self.quantizer_x=Quantizer("Conv2d","input",bit_type=self.bit_type)
        self.quantizer_b=Quantizer("Conv2d","bias",   bit_type="int-16")
        self.quantizer_y=Quantizer("Conv2d","output",bit_type=self.bit_type)

        self.input_shape= None
        self.output_shape=None

    def celibration(self):
        self.quantizer_x.celibration()
        self.quantizer_w.celibration()
        self.quantizer_b.celibration()
        self.quant=True

    def forward(self,x):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
        else:
            y=F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
            
            self.quantizer_w.update_param(self.weight)
            self.quantizer_b.update_param(self.bias)
            self.quantizer_x.update_param(x)
            self.quantizer_y.update_param(y)
            if self.input_shape is None or self.output_shape is  None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y

    def _forward_qnet(self, x):
        int_x       = self.quantizer_x.quant(x)
        fp_x        = self.quantizer_x.dequant(int_x)
        
        int_weight  = self.quantizer_w.quant(self.weight)
        fp_weight   = self.quantizer_w.dequant(int_weight)

        int_bias    = self.quantizer_b.quant(self.bias)
        fp_bias     = self.quantizer_b.dequant(int_bias)

        y = F.conv2d(fp_x,fp_weight,fp_bias,self.stride, self.padding,
                self.dilation, self.groups)  
        
        int_y       = self.quantizer_y.quant(y)    
        fp_y        = self.quantizer_y.dequant(int_y)

        return fp_y

    def _forward_onet(self, x):
        int_x       = self.quantizer_x.quant(x)

        int_weight  = self.quantizer_w.quant(self.weight)
        
        self.quantizer_b.scaler= self.quantizer_w.scaler*self.quantizer_x.scaler

        int_bias    = self.quantizer_b.quant(self.bias)

        int_y = F.conv2d(int_x,int_weight,int_bias,self.stride, self.padding,
                self.dilation, self.groups)  
        
        int_y = (int_y*self.G_value).round().clip(min=self.quantizer_y.bit_lower_bound,
                    max=self.quantizer_y.bit_upper_bound)
          
        fp_y        = self.quantizer_y.dequant(int_y)

        
        return fp_y

    def _forward_intinfer(self,x):
        pass
      
            
        
