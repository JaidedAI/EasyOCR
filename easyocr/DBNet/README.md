# DBNet - Inference Only
This text detection module is adapted from [DBNet++](https://github.com/MhLiao/DB).

## 1. Overview
DBNet works as an image segmentation which performs classification at pixel-level. The model classifies if each pixel from the input image is a part of a text region. This module uses dynamic import and class construction from a config file. Config files are expected to be found in `./configs/`. At the input, the input image is expected to have width and height as multiple of 32. Input images that does not have these dimension will be resized accordingly. In addition, minimum and maximum sizes can be specified in the config file.

### 1.1) Terminology
  * Probability Heatmap: A tensor represents classification confidence of each pixel for being a part of a text region.
  * Segmentation: A boolean-like tensor represents region that is determined as being a text region.
  * text_threshold: A threshold for each element of the probability heatmap to be considered as a text region.
  * detection_size: This term is used to refer to the size of the image on which the detection routine will be performed. Input images that are not of this size will be resized accordingly.

### 1.2) Changes from the original repo
  1. Scripts inside `./concerns/` and multiple `.yaml` files are consolidated and pruned for inference-only implementation and dependencies reduction.
  2. DCN operators, which are required to be compiled with Ahead-of-Time (AoT) in the original repo, are changed to compile with Just-in-Time (JIT) approach as the default. AoT approach is still support.
  3. DCN CPU version is provided in addition to the CUDA version from the original repo.
  4. Pretrained weights are renamed for easy referring and adding file extension.
   
  | Original name                                       | New name                  |
  |-----------------------------------------------------|---------------------------|
  |synthtext_finetune_ic15_res18_dcn_fpn_dbv2           |pretrained_ic15_resnet18.pt|
  |synthtext_finetune_ic15_res50_dcn_fpn_dbv2_thresh0.25|pretrained_ic15_resnet50.pt|
  

## 2. Using and Compiling DCN operators
DBNet requires DCN operators to be compiled. There are two versions of DCN provided; CPU version and CUDA version (original). CUDA version works significantly faster, but requires CUDA-support GPU and CUDA developer toolkit. The CPU version can work without GPU and CUDA. The compilation prerequisites and instruction can be found below.

Please not that, EasyOCR **can work** without DBNet and DCN operators by using CRAFT text detection (the default detector module).

### 2.1) Prerequisites
##### CPU version
 * GCC compiler > 4.9

##### CUDA version
 * GCC compiler > 4.9
 * [CUDA Developer Toolkits](https://developer.nvidia.com/cuda-toolkit) > 9.0 (Tested on 11.3). 

### 2.2) Installing Dependencies

Some step-by-step procedure to install the prerequisites is listed below. Please note that there are other methods that work as well. These methods are listed only to serve as a guideline.

#### Installing GCC Compiler
*Step 1*: Check if your machine already has GCC installed.

On command line terminal (Linux/Mac/Windows);
```
> gcc --version
```
If you already have GCC installed, it will report the version of GCC on your machine. If the command gives an error message along the line of command not found, it implies you do not have GCC installed. 

*Step 2*: Install GCC.

To install GCC, you can do one of the following commands, depending on the privileges of your user account on your machine (Linux/Debian/Ubuntu)
```
> apt-get install build-essential
```
or
```
> sudo apt-get install build-essential
```
For Mac and Windows users, please follow the respective official instructions.

*Step 3*: Verification
Repeat Step 1 to make sure that you now have GCC installed.

#### Installing CUDA and NVCC Compiler
*Step 1*: Check if your machine already has NVCC and CUDA toolkit installed.
On command line terminal (Linux/Mac/Windows);
```
> nvcc --version
```
If you already have NVCC installed, it will report the NVCC version on your machine. If the command gives an error message along the line of command not found, it implies you do not have NVCC installed.

*Step 2*: Install NVCC and CUDA developer toolkit.

Option 1: The official instruction can be found [here](https://developer.nvidia.com/cuda-downloads).

Option 2:
Alternatively, you can try install NVCC with [conda](https://docs.conda.io/projects/conda/en/latest/index.html) (package management system and environment management system).

To use conda to install NVCC, you can do;
*Linux/Mac/Windows*
```
> conda install -c conda-forge cudatoolkit-dev 
```
Note that the above command may fail if your machine is missing some library, such as libxml2. If such error occurs, please install the missing libraries and try again.

*Step 3*: Verification
Repeat Step 1 to make sure that you now have NVCC installed.

#### Installing conda
Step 1: Check if your machine already has conda installed.
On command line terminal, (Linux/Mac/Windows)
```
> conda --version
```
If you already have conda installed, it will report the version of conda on your machine. If the command gives an error message along the line of command not found, it implies you do not have conda installed.

Step 2: Install conda
Please follow the [official instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it according to your OS.

Step 3: Verification
Repeat Step 1 to make sure that you now have conda installed.

#### Using Docker image
For Docker users, please use development level images. For example, pytorch/pytorch:x.xx.x-cudax.x-cudnnx-devel. You can verify if all prerequisites are provided by the image by checking
```
gcc --version
```
and
```
nvcc --version
```

If you already have GCC/NVCC installed, each command will report the version of GCC/NVCC on your machine. If the command gives an error message along the line of command not found, it implies you do not have GCC/NVCC installed. 

### 2.3 Compiling DCN Just-in-Time (JiT)
Once all of the prerequisites have been installed, you can start using `dbnet18` as the detect_network for EasyOCR. The module will compile the source codes automatically when needed. The compilation may take a while if the modules are being loaded and compiled for the first time.

### 2.4 Compiling DCN Ahead-of-Time (AoT)
You can also try compiling DCN with Ahead-of-Time approach. The following procedure may serve as a guideline.

#### 2.4.1 Locate EasyOCR and DBNet module inside it
Start python console environment of your choice, such as Jupyter notebook and Spyder IDE. You can also start one from command line interface (Linux/Mac terminal, etc.) by calling `python` or `python3`;

In python console environment;
```
> import os
> import easyocr
> print(os.dirname(easyocr.__file__))
```
This should show the installation location of easyocr on your machine.

The exact output of the above command depends on many factors and will be likely unique for each user, especially the `username`. As an example, let's assuming the command above returns something like;
```
> /home/username/anaconda3/lib/python3.8/site-packages/easyocr
```

We want to go into the directory where `DBNet` and the DCN source files are located within EasyOCR which can be done by appending `DBNet/assets/ops/dcn` to the path obtained above. For example;
```
/home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet/assets/ops/dcn
```
Access the above directory with any File Manager app on your machine of your choice, for example, Explorer (Windows), Nautilus (Linux/GNOME), Finder (MAC). Or use the following command in the command line interface;
```
> cd /home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet/assets/ops/dcn
```

#### 2.4.2 Compiling DCN operator manually with setup.py script

First go to DCN operator subdirectory inside DBNet module inside EasyOCR directory (e.g. `/home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet/assets/ops/dcn`) by;

Verify that a script `setup.py` is found in that directory. (This version of `setup.py` script is different from the original version from [DBNet++](https://github.com/MhLiao/DB) since the support for CPU has been added.) Once the script is located, run the following command;
```
> python setup.py build_ext --inplace
```
This will start the compilation process and you can monitor the progress, including error messages, if any, on the command line interface. If there is any error, please resolve them, and try again. Once the compilation has been completed, new files will be added to the current directory (i.e. `/home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet/assets/ops/dcn`). If your machine has only CPU, but no CUDA device (GPU), two files will be added to the directory. **Please note that the exact names of the files will be different depending on the configuration of your machine.** The file names should look like;
```
deform_conv_cpu.******.so
deform_pool_cpu.******.so
```
If your machine also has CUDA device, two additional files will be added (4 files in total). The file names of these files should look like;
```
deform_conv_cuda.******.so
deform_pool_cuda.******.so
```

### 3. Using DBNet Detector
When initializing EasyOCR with DBNet as the detect network for the first time in the current working session, messages will be print to indicate if the DCN operators are loaded from objects compiled with AoT approach (pre-compiled) or the source codes are compiling with JiT approach. 




