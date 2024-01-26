import torch
import torch.nn as nn
from ..quant.quantizer import Quantizer

class Hooks():
    def __init__(self,):
        self._id=-1

    def update(self):
        pass

    @property
    def id(self):
        self._id+=1
        return self._id
    


class BaseOps(nn.Module):
    def __init__(self,ops,ops_type,bit_type) -> None:
        super().__init__()
        """
            args:
                ops: torch.nn.ops 
                    the ops in package torch.nn,like nn.Linear,nn.Conv2d..
                ops_type: str
                    optional arg:
                        qnet: evaluate 
                        onet,
                        intinfer(now called onet_nn_infer in relate files,will merge in this ops in the future)
                bit_type: str
                    quant ops to target bit_type
                    warning: conist of sign str and bit width with str '-', for example:int-8,uint-16
                quant: bool
                    contral quantilize or eval
                quantizer: class Quantizer in quant/quantizer.py
                    to quant input/output and param of ops    
                
        """
        self.ops_name="BaseOps"
        self.quant_ops_type =ops_type
        self.int_infer_flag=False
        
        self.quantizer_x=Quantizer("BaseOps","intput",bit_type=bit_type)

        self.input_shape=None
        self.output_shape=None

    def forward(self,x):
        if self.ops_type =='qnet':
            return self._forward_qnet(x)
        elif self.ops_type =='onet':
            return self._forward_onet(x)
        elif self.ops_type == 'intinfer':
            return self._forward_intinfer(x)
        else:
            raise ValueError(f"unsupport quant ops {self.ops_type}")
        
    def _forward_qnet(self,x):
        pass

    def _forward_onet(self,x):
        pass

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