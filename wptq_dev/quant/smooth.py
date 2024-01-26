from transformers.models.opt.modeling_opt import OPTDecoderLayer
from wptq_dev.ddp_nn import DDPLinearAllreduce
import torch.nn.functional as F
import torch

def smooth_lm(model,name='opt'):
    if name=='opt':
        for name, module in model.named_modules():
            if isinstance(module, OPTDecoderLayer):
                attn_ln = module.self_attn_layer_norm
                qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
                smooth_ln_fcs(qkv,attn_ln)
                
                # smooth_ln_fcs(module.self_attn.out_proj)

                # smooth_ln_fcs(module.fc2)

                ffn_ln = module.final_layer_norm
                fc1 = module.fc1
                smooth_ln_fcs(fc1,ffn_ln)
    elif name=='llama2':
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        for name, module in model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                fc1 = module.mlp.down_proj
                smooth_ln_fcs(fc1)
                
    elif name=='vit':
        for name, module in model.named_modules():
            from vit.modeling_vit import ViTOutput
            if isinstance(module, ViTOutput):
                fc = [module.dense]
                smooth_ln_fcs(fc)

    
def smooth_ln_fcs(fcs,ln=None, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    
    act_scales=None

    for fc in fcs:
        if act_scales is None:
            act_scales=fc.quantizer_x.act_scales
        else:
            assert torch.equal(fc.quantizer_x.act_scales, act_scales)

        # assert isinstance(fc, QuantLinear) 
        assert fc.weight.shape[1] == act_scales.numel()

    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5).to(device).to(dtype)
    
    for fc in fcs:
        setattr(fc.quantizer_x,'smooth_factor',scales)
        fc.weight.detach().mul_(scales.view(1, -1))

def smooth_forward(x,weight,bias=None,smooth_factor=None):
    if smooth_factor is not None:
        x_div=x/smooth_factor
    else:
        x_div=x

    scales = x_div.abs().max()
    q_max = 127
    scales.clamp_(min=1e-5).div_(q_max)
    fp_x=scales*torch.round((x_div/scales)).to(weight.dtype)

    scales = weight.abs().max()
    q_max = 127
    scales.clamp_(min=1e-5).div_(q_max)
    fp_weight=scales*torch.round(weight/scales)

    fp_y=F.linear(fp_x, fp_weight,bias)

    return fp_y


def get_smooth_scale():
    pass