import torch.nn as nn
import torch
from ..quant.quantizer import Quantizer

class QuantMatMul(nn.Module):
    def __init__(self,
                bit_type="int-8",
                ops_type='qnet',
            ):
        super().__init__()
        self.ops_name = "MatMul"
        self.ops_type = ops_type
        self.bit_type = bit_type
        

        self.quant=False
        self.quantizer_x1 = Quantizer(self.ops_name,"input",bit_type=self.bit_type,)
        self.quantizer_x2 = Quantizer(self.ops_name,"input",bit_type=self.bit_type)
        self.quantizer_y  = Quantizer(self.ops_name,"output",bit_type=self.bit_type)

        # for onet ops
        self.input1_shape=None
        self.input2_shape=None
        self.output_shape=None 
        self.G_value=None

    def celibration(self,):
        self.quant=True
        self.quantizer_x1.celibration()
        self.quantizer_x2.celibration()
        self.quantizer_y.celibration()
        self.G_value=torch.round(1/(self.quantizer_x1.scaler*self.quantizer_x2.scaler/self.quantizer_y.scaler))
        
    def forward(self,x1,x2):
        if self.quant:
            if self.ops_type=='qnet':
                return self._forward_qnet(x1,x2)
            elif self.ops_type=='onet':
                return self._forward_onet(x1,x2)
            elif self.ops_type == 'intinfer':
                return self._forward_intinfer(x1,x2)
            else:
                raise ValueError
        else:
            y=x1@x2
            self.quantizer_x1.update_param(x1)
            self.quantizer_x2.update_param(x2)
            self.quantizer_y.update_param(y)
            if self.input1_shape is None or self.input2_shape is None \
                or self.output_shape is None:
                    self.input1_shape = x1.size()
                    self.input2_shape = x2.size()
                    self.output_shape = y.size()
            return y
    
    def _forward_qnet(self,x1,x2):
        x1_int=self.quantizer_x1.quant(x1).to(torch.float32)
        x2_int=self.quantizer_x2.quant(x2).to(torch.float32)

        x1_fp =self.quantizer_x1.dequant(x1_int).to(torch.float32)
        x2_fp =self.quantizer_x2.dequant(x2_int).to(torch.float32)
            
        y=x1_fp@x2_fp

        int_y = self.quantizer_y.quant(y)
        fp_y  = self.quantizer_y.dequant(int_y)

        # info={'x1':(x1_fp,x1),'x2':(x2_fp,x2),'y':(fp_y,y)}
        # self.log_info(info)
        return fp_y
    
    def _forward_onet(self,x1,x2):
        x1_int=self.quantizer_x1.quant(x1).to(torch.float32)
        x2_int=self.quantizer_x2.quant(x2).to(torch.float32)
        
        y_int=(x1_int@x2_int/self.G_value).round().clip(min=self.quantizer_y.bit_lower_bound,
                            max=self.quantizer_y.bit_upper_bound)
        
        y_fp = self.quantizer_y.dequant(y_int)

        return y_fp
    
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
    

        

