import torch
from ..nn import *
from ..nn_intinfer import *
import os
from onnxsim import simplify
import onnx
os.environ['ONNX_DISABLE_PREPROCESSING_DEBUG_INFO'] = '1'

def ops_replace(model,module_name,ops):
    name_list=module_name.split('.')
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)  
    setattr(model, name_list[-1],ops)


def export_infer_net(model:torch.nn.Module):
    for m,n in model.named_modules():
        if isinstance(n,QuantSoftmax):
            ops_replace(model,m,OIntInferSoftmax(n))
        if isinstance(n,QuantLayerNorm):
            ops_replace(model,m,OIntInferLayerNorm(n))   
        if isinstance(n,QuantLinear):
            ops_replace(model,m,OIntInferLinear(n))
        if isinstance(n,QuantGelu):
            ops_replace(model,m,OIntInferGelu(n))
        if isinstance(n,QuantMatMul):
            ops_replace(model,m,OIntInferMatMul(n))
        if isinstance(n,QuantAct):
            ops_replace(model,m,OIntInferAct(n))
        if isinstance(n,QuantElemAdd):
            ops_replace(model,m,OIntInferElemAdd(n))
        if isinstance(n,QuantElemRightShift):
            ops_replace(model,m,OIntInferElemLeftShift(n))

def export_onnx_net(model,path,input_size):
    model.eval()

    save_path=os.path.join(path,"wptq_model.onnx")
    sim_path =os.path.join(path,"wptq_model_simplify.onnx")
    x=torch.randn(input_size).cuda()
  
    torch.onnx.export(model,
                      x,
                      save_path,
                      input_names=["inputs"],
                      output_names=["outputs"],
                      dynamic_axes={
                        "input" : {0:"batch_size"},		
                        "output": {0:"batch_size"}
                        },
                    )
   
    onnx_model = onnx.load(save_path)
    simplified_model, _ = simplify(onnx_model)
    
    onnx.save_model(simplified_model, sim_path)