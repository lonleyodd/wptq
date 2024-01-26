import torch.nn as nn
from torch import Tensor
from ..quant.quantizer import Quantizer
import  torch.nn.functional  as F
import torch

class QuantLinear(nn.Linear):
    def __init__(self,
                ops:nn.Linear,
                ops_type='qnet',
                **kwargs
            ):
        super(QuantLinear, self).__init__(
            in_features=ops.in_features,
            out_features=ops.out_features, 
        )
        self.ops_name = 'Linear'
        self.ops_type = ops_type

        self.weight   = ops.weight
        if ops.bias is not None:
            self.bias = ops.bias
        else:
            self.bias = nn.Parameter(torch.zeros(self.weight.size(0),dtype=self.weight.dtype, device=self.weight.device))

        self.quant = False
        self.quantizer_x=Quantizer(self.ops_name,"input",**kwargs["input"])
        self.quantizer_y=Quantizer(self.ops_name,"output",**kwargs["output"])
        self.quantizer_b=Quantizer(self.ops_name,"bias",**kwargs["bias"])
        self.quantizer_w=Quantizer(self.ops_name,"weight",**kwargs["weight"])
       
        # for onet ops
        self.input_shape=None
        self.output_shape=None
        self.G_value = None

    def celibration(self,):
        self.quantizer_x.celibration()
        self.quantizer_w.celibration()
        self.quantizer_b.celibration()
        self.quantizer_y.celibration()

        if hasattr(self.quantizer_x,'smooth_factor'):
           
            #for llm
            # factor= torch.round(self.quantizer_x.smooth_factor)
            # self.quantizer_x.scaler=(self.quantizer_x.scaler.to(factor.device)/factor).max()
            # self.quantizer_w.scaler=(self.quantizer_w.scaler.to(factor.device)*factor).max()

            #for vit
            factor = self.quantizer_x.smooth_factor
            self.quantizer_x.scaler=(self.quantizer_x.scaler.to(factor.device)/factor).max()
            self.quantizer_w.scaler=(self.quantizer_w.scaler.to(factor.device)*factor).max()

        if self.ops_type =='onet':
            self.G_value = torch.round(self.quantizer_y.scaler.to(self.quantizer_x.scaler.device)/(self.quantizer_x.scaler*self.quantizer_w.scaler))
            self.quantizer_b.scaler= self.quantizer_x.scaler*self.quantizer_w.scaler

        self.quant=True

    # def _forward_smooth(self,x):
    #     if self.ops_type=='qnet':
    #         if hasattr(self.quantizer_x,'smooth_factor'):
    #             x=x/self.quantizer_x.smooth_factor
            
    #         x_int = self.quantizer_x.quant(x)
    #         x_fp = self.quantizer_x.dequant(x_int).to(torch.float16)
            
    #         w_int = self.quantizer_w.quant(self.weight)
    #         w_fp = self.quantizer_w.dequant(w_int).to(torch.float16)

    #         b_int   = self.quantizer_b.quant(self.bias)
    #         b_fp    = self.quantizer_b.dequant(b_int)

    #         y=F.linear(x_fp,w_fp,b_fp)

    #         y_int=self.quantizer_y.quant(y)
    #         y_fp=self.quantizer_y.dequant(y_int).to(torch.float16)
            
    #         return y_fp

    #         # info={'x':(x,x_fp),'weight':(self.weight,w_fp),'bias':(self.bias,b_fp),'y':(y,y_fp)}
    #         # self.log_info(info)
    #     else:
    #         if hasattr(self.quantizer_x,'smooth_factor'):
    #             x=x/self.quantizer_x.smooth_factor
            
    #         x_int = self.quantizer_x.quant(x).to(torch.float64)
            
    #         w_int = self.quantizer_w.quant(self.weight).to(torch.float64)

    #         self.quantizer_b.scaler= self.quantizer_x.scaler*self.quantizer_w.scaler

    #         b_int = self.quantizer_b.quant(self.bias).to(torch.float64)
            
    #         y_int = F.linear(x_int,w_int,b_int).to(torch.float64)

    #         y_int = (y_int/self.G_value.to(x.device,dtype=torch.float64)).round().clip(min=self.quantizer_y.bit_lower_bound,
    #                         max=self.quantizer_y.bit_upper_bound)

    #         y_fp  = self.quantizer_y.dequant(y_int).to(torch.float16)
    #         return y_fp
        

    def forward(self,x):
        if self.quant:
            if self.ops_type =='qnet':
                return self._forward_qnet(x)
            elif self.ops_type =='onet':
                return self._forward_onet(x)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x)
        else:
            y=F.linear(x,self.weight,self.bias) 
            self.quantizer_x.update_param(x)
            self.quantizer_w.update_param(self.weight)
            self.quantizer_b.update_param(self.bias)
            self.quantizer_y.update_param(y)

            if self.input_shape is None or self.output_shape is None:
                self.input_shape=x.size()
                self.output_shape=y.size()
            return y 
 
    def _forward_qnet(self, x:Tensor):
        # if hasattr(self.quantizer_x,'smooth_factor'):
        #     x=x/self.quantizer_x.smooth_factor
        # scale=None
        if hasattr(self.quantizer_x,'smooth_factor'): 
            int_x      = self.quantizer_x.quant(x,scale=self.quantizer_x.smooth_factor*self.quantizer_x.scaler)
        else:
            int_x      = self.quantizer_x.quant(x)
            
        fp_x       = self.quantizer_x.dequant(int_x)
        
        int_weight = self.quantizer_w.quant(self.weight)
        fp_weight  = self.quantizer_w.dequant(int_weight)
        
        int_bias   = self.quantizer_b.quant(self.bias)
        fp_bias    = self.quantizer_b.dequant(int_bias)

        y=F.linear(fp_x, fp_weight,fp_bias)
        
        int_y=self.quantizer_y.quant(y)
        fp_y=self.quantizer_y.dequant(int_y)

        info={'x':(x,fp_x),'weight':(self.weight,fp_weight),'bias':(self.bias,fp_bias),'y':(y,fp_y)}
        self.log_info(info)
        return fp_y
    
    def _forward_onet(self,x):
        # if hasattr(self.quantizer_x,'smooth_factor'):
        #     x=x/self.quantizer_x.smooth_factor

        # int_x       = self.quantizer_x.quant(x).to(torch.float32)

        if hasattr(self.quantizer_x,'smooth_factor'): 
            int_x      = self.quantizer_x.quant(x,scale=self.quantizer_x.smooth_factor*self.quantizer_x.scaler).to(torch.float32)
        else:
            int_x      = self.quantizer_x.quant(x).to(torch.float32)
        
        int_bias    = self.quantizer_b.quant(self.bias).to(torch.float32)
        
        int_weight  = self.quantizer_w.quant(self.weight).to(torch.float32)
        
        int_y = F.linear(int_x,int_weight,int_bias).to(torch.float32)
        
        int_y = (int_y/self.G_value).round().clip(min=self.quantizer_y.bit_lower_bound,
                            max=self.quantizer_y.bit_upper_bound)
        
        fp_y  = self.quantizer_y.dequant(int_y).to(torch.float16)

        return fp_y
        
    def log_info(self,info):
        self.logger.debug(f"module name: {self.module_name}, ops id: {self.ops_id}")
        self.logger.debug('------------------------------------------')
        for k,v in info.items():
            if k=="G_value":
                self.logger.debug(f"{self.ops_name} {k} : {v.item()}")
            else:
                quant_x,x=v[0],v[1]
                loss=torch.nn.MSELoss()(quant_x,x)
                self.logger.debug(f"{self.ops_name} quant-dequant {k} loss: {loss.item()}")
        self.logger.debug('------------------------------------------\n')