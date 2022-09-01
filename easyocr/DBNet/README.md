# DBNet - Inference Only
This text detection module is adapted from [DBNet++](https://github.com/MhLiao/DB).

## Overview
DBNet works as an image segmentation which performs classification at pixel-level. The model classifies if each pixel from the input image is a part of a text region. This module uses dynamic import and class construction from a config file. Config files are expected to be found in `./configs/`. At the input, the input image are expected to have width and height as multiple of 32. Input images that does not have these dimension will be resized accordingly. In addition, minimum and maximum sizes can be specified in the config file. *Currently, DBNet text detector requires running with GPU.*

### Terminology
  * Probability Heatmap: A tensor represent classification confidence of each pixel for being a part of text region.
  * Segmentation: A boolean-like tensor represent region that is determined as being a part of a text region.
  * text_threshold: A threshold for each element of the probability heatmap to be consider as text region.
  * detection_size: This term is used to refer to the size of the image on which the detection routine will be performed. Input images that are not of this size will be resized accordingly.

### Compiling DCN operator
DBNet requires DCN operator to be compiled with GPU. The instruction from the original repo can be found [here](https://github.com/MhLiao/DB#requirements). If EasyOCR is install via `pypi`, this process should be done automatically. If the operator is compiled successfully, a flag `dcn_compiling_success` will be added to `./DBNet/`. If the compilation failed during installation, the flag will be missing. Although, EasyOCR **can work** without DBNet and DCN operator by using CRAFT text detection (default).

### Changes from the original repo
  1. Scripts inside `./concerns/` and multiple `.yaml` files are consolidated and pruned for inference-only implementation and dependencies reduction.
  2. A flag file `dcn_compiling_success` is added to indicate when DCN operator is compiled successfully. In addition, a log file `log.txt` is created to collect warning and error messages from compilation process.
  3. Pretrained weights are renamed for easy referring and adding file extension.
  | Original name                                       | New name                  |
  |-----------------------------------------------------|---------------------------|
  |synthtext_finetune_ic15_res18_dcn_fpn_dbv2           |pretrained_ic15_resnet18.pt|
  |synthtext_finetune_ic15_res50_dcn_fpn_dbv2_thresh0.25|pretrained_ic15_resnet50.pt|
  
### Troubleshoot
If DCN operators are failed to compile. You can try compile it manually. The following procedure may serve as a guideline.

#### Locate EasyOCR and DBNet module inside it
In python console environment (Linux/Mac terminal, Jupyter notebook, Spyder IDE, etc.);
```
> import os
> import easyocr
> print(os.dirname(easyocr.__file__))
```
This should show the location of easyocr on the machine. 

The exact output of the above command depends on many factors and will be likely unique for each user, especially the `username`. For the sake of troubleshooting, let's assuming the command above returns something like;
```
/home/username/anaconda3/lib/python3.8/site-packages/easyocr
```

#### Check for error messages from previous compilation

We want to go into the directory where `DBNet` and the compile script are located within EasyOCR which can be done by appending `DBNet` to the path obtained above. For example;
```
> cd /home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet
```
If the compilation had been attempted, there would be a log file `log.txt` in this directory. You can open the log file to check for any error and resolve it. If the compilation hadn't been attempted before, the file would be missing (this is expected).

#### Compiling DCN operator manually

Once the error, if any, from the previous compilation attempt has been resolved, you can try compile the operator again manual. To do so, first go to EasyOCR directory (e.g. `/home/username/anaconda3/lib/python3.8/site-packages/easyocr`) by;
```
> cd /home/username/anaconda3/lib/python3.8/site-packages/easyocr
```
Then, change the directory to subdirectory `scritps/`, by;
```
> cd scripts
```
To verify if you have the compiling script, you can look for a file `compile_dbnet_dcn.py` under subdirectory `scripts` by;
```
> ls
```
This will list all files and subdirectories in the current directory.

Then, to start the compilation, run;
```
> python compile_dbnet_dcn.py
```
This will start the compilation process. If the compilation is completed successfully, a flag (blank file) `dcn_compiling_success` will be added to `/home/username/anaconda3/lib/python3.8/site-packages/easyocr/DBNet`. If the compilation failed, a new `log.txt` will be written and you can check for any other error messages, resolve them, and try again.
