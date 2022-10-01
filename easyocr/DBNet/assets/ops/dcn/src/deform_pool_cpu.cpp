/*
Created by Jaided AI
Released Date: 31/08/2022
Description:
Deformable convolution operator for CPU. 
This code is adapted from;
https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/src/deform_pool_cuda.cpp
https://github.com/CharlesShang/DCNv2
https://github.com/lbin/DCNv2
*/

#include "deform_pool_cpu_kernel.h"
#include <torch/extension.h>
#include <cmath>
#include <vector>

void deform_psroi_pooling_cpu_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, 
    at::Tensor out, at::Tensor top_count, const int no_trans, 
    const float spatial_scale, const int output_dim, const int group_size,
    const int pooled_size, const int part_size, const int sample_per_part,
    const float trans_std) {
  
  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);

  const int num_bbox = bbox.size(0);
  
  if (num_bbox != out.size(0))
    AT_ERROR("Output shape and bbox number wont match: (%d vs %d).",
             out.size(0), num_bbox);

  DeformablePSROIPoolForward(
      input, bbox, trans, 
      out, top_count, 
      batch, channels, height, width,
      num_bbox, channels_trans, no_trans, 
      spatial_scale, output_dim, group_size,
      pooled_size, part_size, sample_per_part, 
      trans_std);
}

void deform_psroi_pooling_cpu_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, 
    at::Tensor trans, at::Tensor top_count, at::Tensor input_grad, 
    at::Tensor trans_grad, const int no_trans, 
    const float spatial_scale, const int output_dim, const int group_size, 
    const int pooled_size, const int part_size, const int sample_per_part, 
    const float trans_std) {
  TORCH_CHECK(out_grad.is_contiguous(), "out_grad tensor has to be contiguous");
  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);

  const int num_bbox = bbox.size(0);
  if (num_bbox != out_grad.size(0))
    AT_ERROR("Output shape and bbox number wont match: (%d vs %d).",
             out_grad.size(0), num_bbox);

  DeformablePSROIPoolBackwardAcc(
      out_grad, input, bbox, 
      trans, top_count, input_grad, trans_grad, 
      batch, channels, height, width, 
      num_bbox, channels_trans, no_trans,
      spatial_scale, output_dim, group_size, 
      pooled_size, part_size, sample_per_part, 
      trans_std);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_psroi_pooling_cpu_forward", 
        &deform_psroi_pooling_cpu_forward,
        "deform psroi pooling forward(CPU)");
  m.def("deform_psroi_pooling_cpu_backward",
        &deform_psroi_pooling_cpu_backward,
        "deform psroi pooling backward(CPU)");
}
