'''
Modified by Jaided AI
Released Date: 31/08/2022
Description:
- Add support for Deformable convolution operator on CPU for forward propagation.
- Change to Just-in-Time loading approach
'''
import os
import torch
import warnings
from torch.autograd import Function
from torch.nn.modules.utils import _pair
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
    from .. import deform_conv_cpu
    warnings.warn("Using precompiled deform_conv_cpu from {}".format(deform_conv_cpu.__file__))
    dcn_cpu_ready = True
except:
    try:
        warnings.warn("Compiling deform_conv_cpu ...")
        warnings.warn("(This may take a while if this module is loaded for the first time.)")
        deform_conv_cpu = cpp_extension.load(
                            name="deform_conv_cpu", 
                            sources=[os.path.join(dcn_dir, 'src', "deform_conv_cpu.cpp"),
                                     os.path.join(dcn_dir, 'src', "deform_conv_cpu_kernel.cpp")])
        warnings.warn("Done.")
        dcn_cpu_ready = True
    except Exception as error:
        warnings.warn(' '.join([
            "Failed to import and/or compile 'deform_conv_cpu' with the following error",
            "{}".format(error),
            "Deformable convulution and DBNet will not be able to run on CPU."
            ]))
        dcn_cpu_ready = False

if torch.cuda.is_available():
    try:
        from .. import deform_conv_cuda
        warnings.warn("Using precompiled deform_conv_cuda from {}".format(deform_conv_cuda.__file__))
        dcn_cuda_ready = True
    except:
        try:
            warnings.warn("Compiling deform_conv_cuda ...")
            warnings.warn("(This may take a while if this module is loaded for the first time.)")
            cuda_sources = [os.path.join(dcn_dir, 'src', src_file) 
                           for src_file in ["deform_conv_cuda.cpp",
                                            "deform_conv_cuda_kernel.cu"]
                           ]
            deform_conv_cuda = cpp_extension.load(
                                name="deform_conv_cuda", 
                                sources=[os.path.join(dcn_dir, 'src', "deform_conv_cuda.cpp"),
                                         os.path.join(dcn_dir, 'src', "deform_conv_cuda_kernel.cu")])
            warnings.warn("Done.")
            dcn_cuda_ready = True
        except Exception as error:
            warnings.warn(' '.join([
                "Failed to import or compile 'deform_conv_cuda' with the following error",
                "{}".format(error),
                "Deformable convulution and DBNet will not be able to run on GPU."
                ]))
            dcn_cuda_ready = False

class DeformConvFunction(Function):
    
    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConvFunction._output_size(input, weight, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.shape[0])
        assert (input.shape[0] %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'
        if not input.is_cuda and dcn_cpu_ready:
            deform_conv_cpu.deform_conv_forward_cpu(
                input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                cur_im2col_step)
        elif input.is_cuda and dcn_cuda_ready:
            deform_conv_cuda.deform_conv_forward_cuda(
                input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                cur_im2col_step)
        else:
            device_ = input.device.type
            raise RuntimeError(
                "Input type is {}, but 'deform_conv_{}.*.so' is not imported successfully.".format(device_, device_),
                )
             
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError("DCN operator for cpu for backward propagation is not implemented.")
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, ctx.bufs_[0], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                    cur_im2col_step)

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1,
                    cur_im2col_step)

        return (grad_input, grad_offset, grad_weight, None, None, None, None,
                None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda and dcn_cpu_ready:
            deform_conv_cpu.modulated_deform_conv_cpu_forward(
                input, weight, bias, ctx._bufs[0], offset, mask, output,
                ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride,
                ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                ctx.groups, ctx.deformable_groups, ctx.with_bias)
        elif input.is_cuda and dcn_cuda_ready:
            deform_conv_cuda.modulated_deform_conv_cuda_forward(
                input, weight, bias, ctx._bufs[0], offset, mask, output,
                ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride,
                ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                ctx.groups, ctx.deformable_groups, ctx.with_bias)
        else:
            device_ = input.device.type
            raise RuntimeError(
                "Input type is {}, but 'deform_conv_{}.*.so' is not imported successfully.".format(device_, device_),
                )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError("DCN operator for CPU for backward propagation is not implemented.")
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1],
            grad_input, grad_weight, grad_bias, grad_offset, grad_mask,
            grad_output, weight.shape[2], weight.shape[3], ctx.stride,
            ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
            ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply
