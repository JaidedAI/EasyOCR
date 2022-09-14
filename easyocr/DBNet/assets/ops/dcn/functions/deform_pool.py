'''
Modified by Jaided AI
Released Date: 31/08/2022
Description:
- Add support for Deformable convolution operator on CPU for forward propagation.
- Change to Just-in-Time loading approach
'''
import os
import warnings
import torch
from torch.autograd import Function
from torch.utils import cpp_extension

# TODO - Jaided AI: 
# 1. Find a better way to handle and support both Ahead-of-Time (AoT) and Just-in-Time (JiT) compilation.
# 2. Find a better way to report error to help pinpointing issues if there is any.
# Note on JiT and AoT compilation:
# This module supports both AoT and JiT compilation approaches. JiT is hardcoded as the default. If AoT compiled objects are present, it will supercede JiT compilation.
 
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning
dcn_dir = os.path.dirname(os.path.dirname(__file__))
try:
    from .. import deform_pool_cpu
    warnings.warn("Using precompiled deform_pool_cpu from {}".format(deform_pool_cpu.__file__))
    dcn_cpu_ready = True
except:
    try:
        warnings.warn("Compiling deform_pool_cpu ...")
        warnings.warn("(This may take a while if this module is loaded for the first time.)")
        deform_pool_cpu = cpp_extension.load(
                            name="deform_pool_cpu", 
                            sources=[os.path.join(dcn_dir, 'src', "deform_pool_cpu.cpp"),
                                     os.path.join(dcn_dir, 'src', "deform_pool_cpu_kernel.cpp")])
        warnings.warn("Done.")
        dcn_cpu_ready = True
    except Exception as error:
        warnings.warn(' '.join([
            "Failed to import or compile 'deform_pool_cpu' with the following error",
            "{}".format(error),
            "Deformable convulution and DBNet will not be able to run on CPU."
            ]))
        dcn_cpu_ready = False

if torch.cuda.is_available():
    try:
        from .. import deform_pool_cuda
        warnings.warn("Using precompiled deform_pool_cuda from {}".format(deform_pool_cuda.__file__))
        dcn_cuda_ready = True
    except:
        try:
            warnings.warn("Compiling deform_pool_cuda ...")
            warnings.warn("(This may take a while if this module is loaded for the first time.)")
            deform_pool_cuda = cpp_extension.load(
                                name="deform_pool_cuda", 
                                sources=[os.path.join(dcn_dir, 'src', "deform_pool_cuda.cpp"),
                                         os.path.join(dcn_dir, 'src', "deform_pool_cuda_kernel.cu")])
            warnings.warn("Done.")
            dcn_cuda_ready = True
        except Exception as error:
            warnings.warn(' '.join([
                "Failed to import or compile 'deform_pool_cuda' with the following error",
                "{}".format(error),
                "Deformable convulution and DBNet will not be able to run on GPU."
                ]))
            dcn_cuda_ready = False

class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                data,
                rois,
                offset,
                spatial_scale,
                out_size,
                out_channels,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        if not data.is_cuda and dcn_cpu_ready:
            deform_pool_cpu.deform_psroi_pooling_cpu_forward(
                data, rois, offset, output, output_count, ctx.no_trans,
                ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size,
                ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        elif data.is_cuda and dcn_cuda_ready:    
            deform_pool_cuda.deform_psroi_pooling_cuda_forward(
                data, rois, offset, output, output_count, ctx.no_trans,
                ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size,
                ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        else:
            device_ = input.device.type
            raise RuntimeError(
                "Input type is {}, but 'deform_conv_{}.*.so' is not imported successfully.".format(device_, device_),
                )
        
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError("DCN operator for cpu for backward propagation is not implemented.")

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)

        deform_pool_cuda.deform_psroi_pooling_cuda_backward(
            grad_output, data, rois, offset, output_count, grad_input,
            grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels,
            ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part,
            ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
                None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply
