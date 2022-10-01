/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

/*
Modified by Jaided AI
Released Date: 31/08/2022
Description:
Deformable convolution kernel for CPU. 
This code is adapted from;
https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/src/deform_conv_cuda_kernel.cu
https://github.com/CharlesShang/DCNv2
https://github.com/lbin/DCNv2
*/

#include "deform_conv_cpu_kernel.h"
#include <ATen/ATen.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

float deformable_im2col_bilinear(
        const float *bottom_data, const int data_width,
        const int height, const int width, 
        float h, float w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

float get_gradient_weight(
        float argmax_h, float argmax_w,
        const int h, const int w, 
        const int height, const int width) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

float get_coordinate_weight(
        float argmax_h, float argmax_w,
        const int height, const int width, 
        const float *im_data, const int data_width, 
        const int bp_dir) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

void deformable_im2col_cpu_kernel(
        const int n, const float *data_im, const float *data_offset,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int num_channels, const int deformable_group,
        const int height_col, const int width_col,
        float *data_col) {

  for(int index=0; index<n; index++)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void deformable_im2col(
        const float* data_im, const float* data_offset, const int channels,
        const int height, const int width, const int ksize_h, const int ksize_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int parallel_imgs,
        const int deformable_group, float* data_col) {

  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  deformable_im2col_cpu_kernel(
    num_kernels, data_im, data_offset, 
    height, width, ksize_h, ksize_w,
    pad_h, pad_w, stride_h, stride_w, 
    dilation_h, dilation_w, channel_per_deformable_group, 
    parallel_imgs, channels, deformable_group,
    height_col, width_col, data_col);

}

void deformable_col2im_cpu_kernel(
        const int n, const float *data_col, const float *data_offset, const int channels, 
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int deformable_group,
        const int height_col, const int width_col, float *grad_im) {

  for(int index = 0; index < n; index++)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
                                                        2 * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const float cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          float weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;
        }
      }
    }
  }
}

void deformable_col2im(
        const float* data_col, const float* data_offset, const int channels,
        const int height, const int width, const int ksize_h, const int ksize_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int parallel_imgs, const int deformable_group, float* grad_im) {

  // todo: make sure parallel_imgs is passed in correctly
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  deformable_col2im_cpu_kernel(
      num_kernels, data_col, data_offset, channels, 
      height, width, ksize_h, ksize_w, 
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group,
      parallel_imgs, deformable_group, height_col, width_col, grad_im);
      
}

void deformable_col2im_coord_cpu_kernel(
        const int n, const float *data_col, const float *data_im, 
        const float *data_offset, const int channels, 
        const int height, const int width, const int kernel_h, const int kernel_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int offset_channels, const int deformable_group,
        const int height_col, const int width_col, float *grad_offset) {

  for(int index = 0; index < n; index++)
  {
    float val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group *
                                                  batch_size * width_col * height_col;
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
                                                channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 *
                                                        kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
    
      const float weight = get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

void deformable_col2im_coord(
        const float* data_col, const float* data_im, 
        const float* data_offset, const int channels, 
        const int height, const int width, const int ksize_h, const int ksize_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
        const int dilation_h, const int dilation_w, const int parallel_imgs, 
        const int deformable_group, float* grad_offset) {

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group * parallel_imgs;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

  deformable_col2im_coord_cpu_kernel(
      num_kernels, data_col, data_im, 
      data_offset, channels, 
      height, width, ksize_h, ksize_w, 
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group,
      parallel_imgs, 2 * ksize_h * ksize_w * deformable_group, deformable_group,
      height_col, width_col, grad_offset);

}

float dmcn_im2col_bilinear_cpu(
        const float *bottom_data, const int data_width,
        const int height, const int width, 
        float h, float w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

float dmcn_get_gradient_weight_cpu(
        float argmax_h, float argmax_w,
        const int h, const int w, 
        const int height, const int width) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

float dmcn_get_coordinate_weight_cpu(
        float argmax_h, float argmax_w,
        const int height, const int width,
        const float *im_data, const int data_width, 
        const int bp_dir) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

void modulated_deformable_im2col_cpu_kernel(
        const int n, const float *data_im, const float *data_offset, const float *data_mask,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int num_channels, const int deformable_group,
        const int height_col, const int width_col, float *data_col) {

  for(int index=0; index<n; index++)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          val = dmcn_im2col_bilinear_cpu(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void modulated_deformable_col2im_cpu_kernel(
        const int n, const float *data_col, const float *data_offset, 
        const float *data_mask, const int channels, 
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int deformable_group,
        const int height_col, const int width_col, float *grad_im) {

  for(int index = 0; index < n; index++)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const float cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          float weight = dmcn_get_gradient_weight_cpu(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;

        }
      }
    }
  }
}

void modulated_deformable_col2im_coord_cpu_kernel(
        const int n, const float *data_col, const float *data_im,
        const float *data_offset, const float *data_mask, const int channels, 
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
        const int batch_size, const int offset_channels, const int deformable_group,
        const int height_col, const int width_col, float *grad_offset, float *grad_mask) {

  for(int index = 0; index < n; index++)
  {
    float val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      else
      {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear_cpu(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const float weight = dmcn_get_coordinate_weight_cpu(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}

void modulated_deformable_im2col_cpu(
        const float* data_im, const float* data_offset, const float* data_mask,
        const int batch_size, const int channels, const int height_im, const int width_im, 
        const int height_col, const int width_col, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
        const int dilation_h, const int dilation_w, const int deformable_group, 
        float* data_col) {

  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_cpu_kernel(
      num_kernels, data_im, data_offset, data_mask, 
      height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, 
      dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, 
      height_col, width_col, data_col);
}

void modulated_deformable_col2im_cpu(
        const float* data_col, const float* data_offset, const float* data_mask,
        const int batch_size, const int channels, const int height_im, const int width_im, 
        const int height_col, const int width_col, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
        const int dilation_h, const int dilation_w, const int deformable_group, 
        float* grad_im) {

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  modulated_deformable_col2im_cpu_kernel(
      num_kernels, data_col, data_offset, data_mask, channels, 
      height_im, width_im, kernel_h, kernel_w, 
      pad_h, pad_h, stride_h, stride_w,
      dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, deformable_group, 
      height_col, width_col, grad_im);
}

void modulated_deformable_col2im_coord_cpu(
        const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
        const int batch_size, const int channels, const int height_im, const int width_im, 
        const int height_col, const int width_col, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
        const int dilation_h, const int dilation_w, const int deformable_group,
        float* grad_offset, float* grad_mask) {

  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  modulated_deformable_col2im_coord_cpu_kernel(
        num_kernels, data_col, data_im, 
        data_offset, data_mask, channels, 
        height_im, width_im, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, 
        height_col, width_col, grad_offset, grad_mask);
}
