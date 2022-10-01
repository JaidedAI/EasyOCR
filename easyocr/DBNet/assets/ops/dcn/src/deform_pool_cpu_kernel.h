/*
Created by Jaided AI
Released Date: 31/08/2022
Description:
Deformable convolution kernel for CPU. 
This code is adapted from;
https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/src/deform_pool_cuda_kernel.cu
https://github.com/CharlesShang/DCNv2
https://github.com/lbin/DCNv2
*/

#include <torch/extension.h>
#pragma once
#ifndef DEFORM_POOL_CPU_KERNEL
#define DEFORM_POOL_CPU_KERNEL

void DeformablePSROIPoolForward(
    const at::Tensor data, const at::Tensor bbox, const at::Tensor trans,
    at::Tensor out, at::Tensor top_count, 
    const int batch, const int channels, const int height, const int width, 
    const int num_bbox, const int channels_trans, const int no_trans, 
    const float spatial_scale, const int output_dim, const int group_size, 
    const int pooled_size, const int part_size, const int sample_per_part, 
    const float trans_std);

void DeformablePSROIPoolBackwardAcc(
    const at::Tensor out_grad, const at::Tensor data, const at::Tensor bbox,
    const at::Tensor trans, const at::Tensor top_count, at::Tensor in_grad, at::Tensor trans_grad, 
    const int batch, const int channels, const int height, const int width, 
    const int num_bbox, const int channels_trans, const int no_trans, 
    const float spatial_scale, const int output_dim, const int group_size, 
    const int pooled_size, const int part_size, const int sample_per_part, 
    const float trans_std);

#endif
