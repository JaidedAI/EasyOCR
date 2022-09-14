/*
Created by Jaided AI
Released Date: 31/08/2022
Description:
Deformable convolution kernel for CPU. 
This code is adapted from;
https://github.com/MhLiao/DB/blob/master/assets/ops/dcn/src/deform_conv_cuda.cpp
https://github.com/CharlesShang/DCNv2
https://github.com/lbin/DCNv2
*/

#pragma once
#ifndef DEFORM_CONV_CPU_KERNEL
#define DEFORM_CONV_CPU_KERNEL

void deformable_im2col(
        const float *data_im, const float *data_offset, const int channels, 
        const int height, const int width, const int ksize_h, const int ksize_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int parallel_imgs, 
        const int deformable_group, float *data_col);

void deformable_col2im(
        const float *data_col, const float *data_offset, const int channels, 
        const int height, const int width, const int ksize_h, const int ksize_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int parallel_imgs, 
        const int deformable_group, float *grad_im);

void deformable_col2im_coord(
        const float *data_col, const float *data_im,
        const float *data_offset, const int channels, 
        const int height, const int width, const int ksize_h, const int ksize_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int parallel_imgs,
        const int deformable_group, float *grad_offset);

void modulated_deformable_im2col_cpu(
        const float *data_im, const float *data_offset, const float *data_mask, 
        const int batch_size, const int channels, const int height_im, const int width_im, 
        const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int deformable_group,
        float *data_col);

void modulated_deformable_col2im_cpu(
        const float *data_col, const float *data_offset, const float *data_mask, 
        const int batch_size, const int channels, const int height_im, const int width_im,
        const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int deformable_group,
        float *grad_im);

void modulated_deformable_col2im_coord_cpu(
        const float *data_col, const float *data_im, const float *data_offset, const float *data_mask,
        const int batch_size, const int channels, const int height_im, const int width_im, 
        const int height_col, const int width_col, const int kernel_h, const int kenerl_w, 
        const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
        const int dilation_h, const int dilation_w, const int deformable_group, 
        float *grad_offset, float *grad_mask);

#endif


