import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

modules = [
        CppExtension('deform_conv_cpu', [
            'src/deform_conv_cpu.cpp',
            'src/deform_conv_cpu_kernel.cpp',
        ]),
        CppExtension('deform_pool_cpu', [
            'src/deform_pool_cpu.cpp', 
            'src/deform_pool_cpu_kernel.cpp'
        ])
]

if torch.cuda.is_available():
    modules.extend([
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ]),
        CUDAExtension('deform_pool_cuda', [
            'src/deform_pool_cuda.cpp', 
            'src/deform_pool_cuda_kernel.cu'
        ])
    ])

setup(
    name='deform_conv',
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension})
