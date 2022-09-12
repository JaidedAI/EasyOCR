/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/
// modify from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/cuda/deform_psroi_pooling_cuda.cu

/*
Modified by Jaided AI
Released Date: 31/08/2022
Description:
Deformable convolution kernel for CPU. 
This code is adapted from;
https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/src/deform_pool_cuda_kernel.cu
https://github.com/CharlesShang/DCNv2
https://github.com/lbin/DCNv2
*/

#include <torch/extension.h>
#include "deform_pool_cpu_kernel.h"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <ATen/ATen.h>

template <typename T>
T bilinear_interp_cpu(
    const T *data, const T x, const T y,
    const int width, const int height) {

  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  T dist_x = static_cast<T>(x - x1);
  T dist_y = static_cast<T>(y - y1);
  T value11 = data[y1 * width + x1];
  T value12 = data[y2 * width + x1];
  T value21 = data[y1 * width + x2];
  T value22 = data[y2 * width + x2];
  T value = (1 - dist_x) * (1 - dist_y) * value11 +
            (1 - dist_x) * dist_y * value12 +
            dist_x * (1 - dist_y) * value21 +
            dist_x * dist_y * value22;
  return value;
}

template <typename T>
void DeformablePSROIPoolForwardKernelCpu(
    const int count, const T *bottom_data, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const T *bottom_rois, const T *bottom_trans, 
    const int no_trans, const T trans_std, const int sample_per_part, 
    const int output_dim, const int group_size, const int part_size, 
    const int num_classes, const int channels_each_class, 
    T *top_data, T *top_count) {

  for(int index = 0; index < count; index++)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_start_h = static_cast<T>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    T roi_end_w = static_cast<T>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = static_cast<T>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1)); //avoid 0
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

    int part_h = floor(static_cast<T>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? static_cast<T>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;
    T trans_y = no_trans ? static_cast<T>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;

    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    T sum = 0;
    int count = 0;
    int gw = floor(static_cast<T>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_size / pooled_height);
    gw = std::min(std::max(gw, 0), group_size - 1);
    gh = std::min(std::max(gh, 0), group_size - 1);

    const T *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = std::min(std::max(w, T(0.)), width - T(1.));
        h = std::min(std::max(h, T(0.)), height - T(1.));
        int c = (ctop * group_size + gh) * group_size + gw;
        T val = bilinear_interp_cpu(offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        count++;
      }
    }
    top_data[index] = count == 0 ? static_cast<T>(0) : sum / count;
    top_count[index] = count;
  }
}

template <typename T>
void DeformablePSROIPoolBackwardAccKernelCpu(
    const int count, const T *top_diff, const T *top_count,
    const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int output_dim, 
    T *bottom_data_diff, T *bottom_trans_diff,
    const T *bottom_data, const T *bottom_rois, const T *bottom_trans,
    const int no_trans, const T trans_std, const int sample_per_part,
    const int group_size, const int part_size, const int num_classes,
    const int channels_each_class) {

  for(int index = 0; index < count; index++)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_start_h = static_cast<T>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    T roi_end_w = static_cast<T>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = static_cast<T>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    
    // Force too small ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1)); //avoid 0
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

    int part_h = floor(static_cast<T>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? static_cast<T>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;
    T trans_y = no_trans ? static_cast<T>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;

    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0)
    {
      continue;
    }
    T diff_val = top_diff[index] / top_count[index];
    const T *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
    T *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = floor(static_cast<T>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_size / pooled_height);
    gw = std::min(std::max(gw, 0), group_size - 1);
    gh = std::min(std::max(gh, 0), group_size - 1);

    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = std::min(std::max(w, T(0.)), width - T(1.));
        h = std::min(std::max(h, T(0.)), height - T(1.));
        int c = (ctop * group_size + gh) * group_size + gw;
        // backward on feature
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);
        T dist_x = w - x0, dist_y = h - y0;
        T q00 = (1 - dist_x) * (1 - dist_y);
        T q01 = (1 - dist_x) * dist_y;
        T q10 = dist_x * (1 - dist_y);
        T q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
       *(offset_bottom_data_diff + bottom_index_base + y0 * width + x0) += q00 * diff_val;
       *(offset_bottom_data_diff + bottom_index_base + y1 * width + x0) += q01 * diff_val;
       *(offset_bottom_data_diff + bottom_index_base + y0 * width + x1) += q10 * diff_val;
       *(offset_bottom_data_diff + bottom_index_base + y1 * width + x1) += q11 * diff_val;


        if (no_trans)
        {
          continue;
        }
        T U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        T U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        T U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        T U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
        T diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;
        diff_x *= roi_width;
        T diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;
        diff_y *= roi_height;

        *(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w) += diff_x;
        *(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w) += diff_y;
      }
    }
  }
}

void DeformablePSROIPoolForward(
    const at::Tensor input, const at::Tensor bbox, 
    const at::Tensor trans, at::Tensor out, at::Tensor top_count,
    const int batch, const int channels, const int height, const int width,
    const int num_bbox, const int channels_trans, const int no_trans,
    const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std) {

  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "DeformablePSROIPoolForward", [&] {
    DeformablePSROIPoolForwardKernelCpu<scalar_t>(
        out_size, input.contiguous().data_ptr<scalar_t>(), spatial_scale,
        channels, height, width,
        pooled_height, pooled_width,
        bbox.contiguous().data_ptr<scalar_t>(), 
        trans.contiguous().data_ptr<scalar_t>(),
        no_trans, trans_std, sample_per_part,
        output_dim, group_size, part_size,
        num_classes, channels_each_class,
        out.data_ptr<scalar_t>(), 
        top_count.data_ptr<scalar_t>());

  });

}

void DeformablePSROIPoolBackwardAcc(
    const at::Tensor out_grad, const at::Tensor input, const at::Tensor bbox,
    const at::Tensor trans, const at::Tensor top_count,
    at::Tensor in_grad, at::Tensor trans_grad,
    const int batch, const int channels, const int height, const int width,
    const int num_bbox, const int channels_trans, const int no_trans,
    const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std) {
  // LOG(INFO) << "DeformablePSROIPoolBackward";
  const int num_rois = num_bbox;
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  AT_DISPATCH_FLOATING_TYPES(out_grad.scalar_type(), "DeformablePSROIPoolBackwardAcc", [&] {
    DeformablePSROIPoolBackwardAccKernelCpu<scalar_t>(
        out_size, 
        out_grad.contiguous().data_ptr<scalar_t>(), 
        top_count.contiguous().data_ptr<scalar_t>(),
        num_rois, spatial_scale,
        channels, height, width,
        pooled_height, pooled_width, output_dim,
        in_grad.contiguous().data_ptr<scalar_t>(),
        trans_grad.contiguous().data_ptr<scalar_t>(),
        input.contiguous().data_ptr<scalar_t>(),
        bbox.contiguous().data_ptr<scalar_t>(),
        trans.contiguous().data_ptr<scalar_t>(),
        no_trans, trans_std, sample_per_part,
        group_size, part_size, num_classes, 
        channels_each_class);
  });
 

}
