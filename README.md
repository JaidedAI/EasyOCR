# EasyOCR

[![PyPI Status](https://badge.fury.io/py/easyocr.svg)](https://badge.fury.io/py/easyocr)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.to/easyocr)
[![Tweet](https://img.shields.io/twitter/url/https/github.com/JaidedAI/EasyOCR.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20library:%20EasyOCR%20https://github.com/JaidedAI/EasyOCR)
[![Twitter](https://img.shields.io/badge/twitter-@JaidedAI-blue.svg?style=flat)](https://twitter.com/JaidedAI)

Ready-to-use OCR with 80+ [supported languages](https://www.jaided.ai/easyocr) and all popular writing scripts including: Latin, Chinese, Arabic, Devanagari, Cyrillic, etc.

[Try Demo on our website](https://www.jaided.ai/easyocr)

## What's new
- 11 September 2021 - Version 1.4.1
    - Add trainer folder
    - Add `readtextlang` method (thanks[@arkya-art](https://github.com/arkya-art), see [PR](https://github.com/JaidedAI/EasyOCR/pull/525))
    - Extend `rotation_info` argument to support all possible angles (thanks[abde0103](https://github.com/abde0103), see [PR](https://github.com/JaidedAI/EasyOCR/pull/515))
- 29 June 2021 - Version 1.4
    - [Instructions](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md) on training/using custom recognition models
    - Example [dataset](https://www.jaided.ai/easyocr/modelhub) for model training
    - Batched image inference for GPUs (thanks [@SamSamhuns](https://github.com/SamSamhuns), see [PR](https://github.com/JaidedAI/EasyOCR/pull/458))
    - Vertical text support (thanks [@interactivetech](https://github.com/interactivetech)). This is for rotated text, not to be confused with vertical Chinese or Japanese text. (see [PR](https://github.com/JaidedAI/EasyOCR/pull/450))
    - Output in dictionary format (thanks [@A2va](https://github.com/A2va), see [PR](https://github.com/JaidedAI/EasyOCR/pull/441))
- 30 May 2021 - Version 1.3.2
    - Faster greedy decoder (thanks [@samayala22](https://github.com/samayala22))
    - Fix bug when a text box's aspect ratio is disproportional (thanks [iQuartic](https://iquartic.com/) for bug report)
- 20 April 2021 - Version 1.3.1
    - Add support for PIL image (thanks [@prays](https://github.com/prays))
    - Add Tajik language (tjk)
    - Update argument setting for command line
    - Add `x_ths` and `y_ths` to control merging behavior when `paragraph=True`
- 21 March 2021 - Version 1.3
    - Second-generation models: multiple times smaller size, multiple times faster inference, additional characters and comparable accuracy to the first generation models.
    EasyOCR will choose the latest model by default but you can also specify which model to use by passing `recog_network` argument when creating a `Reader` instance.
    For example, `reader = easyocr.Reader(['en','fr'], recog_network='latin_g1')` will use the 1st generation Latin model
    - List of all models: [Model hub](https://www.jaided.ai/easyocr/modelhub)

- [Read all release notes](https://github.com/JaidedAI/EasyOCR/blob/master/releasenotes.md)

## What's coming next
- Handwritten text support

## Examples

![example](examples/example.png)

![example2](examples/example2.png)

![example3](examples/example3.png)


## Installation

Install using `pip`

For the latest stable release:

``` bash
pip install easyocr
```

For the latest development release:

``` bash
pip install git+git://github.com/jaidedai/easyocr.git
```

Note 1: For Windows, please install torch and torchvision first by following the official instructions here https://pytorch.org. On the pytorch website, be sure to select the right CUDA version you have. If you intend to run on CPU mode only, select `CUDA = None`.

Note 2: We also provide a Dockerfile [here](https://github.com/JaidedAI/EasyOCR/blob/master/Dockerfile).

## Usage

``` python
import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('chinese.jpg')
```

The output will be in a list format, each item represents a bounding box, the text detected and confident level, respectively.

``` bash
[([[189, 75], [469, 75], [469, 165], [189, 165]], '愚园路', 0.3754989504814148),
 ([[86, 80], [134, 80], [134, 128], [86, 128]], '西', 0.40452659130096436),
 ([[517, 81], [565, 81], [565, 123], [517, 123]], '东', 0.9989598989486694),
 ([[78, 126], [136, 126], [136, 156], [78, 156]], '315', 0.8125889301300049),
 ([[514, 126], [574, 126], [574, 156], [514, 156]], '309', 0.4971577227115631),
 ([[226, 170], [414, 170], [414, 220], [226, 220]], 'Yuyuan Rd.', 0.8261902332305908),
 ([[79, 173], [125, 173], [125, 213], [79, 213]], 'W', 0.9848111271858215),
 ([[529, 173], [569, 173], [569, 213], [529, 213]], 'E', 0.8405593633651733)]
```
Note 1: `['ch_sim','en']` is the list of languages you want to read. You can pass
several languages at once but not all languages can be used together.
English is compatible with every language and languages that share common characters are usually compatible with each other.

Note 2: Instead of the filepath `chinese.jpg`, you can also pass an OpenCV image object (numpy array) or an image file as bytes. A URL to a raw image is also acceptable.

Note 3: The line `reader = easyocr.Reader(['ch_sim','en'])` is for loading a model into memory. It takes some time but it needs to be run only once.

You can also set `detail=0` for simpler output.

``` python
reader.readtext('chinese.jpg', detail = 0)
```
Result:
``` bash
['愚园路', '西', '东', '315', '309', 'Yuyuan Rd.', 'W', 'E']
```

Model weights for the chosen language will be automatically downloaded or you can
download them manually from the [model hub](https://www.jaided.ai/easyocr/modelhub) and put them in the '~/.EasyOCR/model' folder

In case you do not have a GPU, or your GPU has low memory, you can run the model in CPU-only mode by adding `gpu=False`.

``` python
reader = easyocr.Reader(['ch_sim','en'], gpu=False)
```

For more information, read the [tutorial](https://www.jaided.ai/easyocr/tutorial) and [API Documentation](https://www.jaided.ai/easyocr/documentation).

#### Run on command line

```shell
$ easyocr -l ch_sim en -f chinese.jpg --detail=1 --gpu=True
```

## Train/use your own model

[Read here](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)

## Implementation Roadmap

- Handwritten support
- Restructure code to support swappable detection and recognition algorithms
The api should be as easy as
``` python
reader = easyocr.Reader(['en'], detection='DB', recognition = 'Transformer')
```
The idea is to be able to plug-in any state-of-the-art model into EasyOCR. There are a lot of geniuses trying to make better detection/recognition models, but we are not trying to be geniuses here. We just want to make their works quickly accessible to the public ... for free. (well, we believe most geniuses want their work to create a positive impact as fast/big as possible) The pipeline should be something like the below diagram. Grey slots are placeholders for changeable light blue modules.

![plan](examples/easyocr_framework.jpeg)

## Acknowledgement and References

This project is based on research and code from several papers and open-source repositories.

All deep learning execution is based on [Pytorch](https://pytorch.org). :heart:

Detection execution uses the CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941) (Thanks @YoungminBaek from @clovaai). We also use their pretrained model.

The recognition model is a CRNN ([paper](https://arxiv.org/abs/1507.05717)). It is composed of 3 main components: feature extraction (we are currently using [Resnet](https://arxiv.org/abs/1512.03385)) and VGG, sequence labeling ([LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)) and decoding ([CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). The training pipeline for recognition execution is a modified version of the [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) framework. (Thanks @ku21fan from @clovaai) This repository is a gem that deserves more recognition.

Beam search code is based on this [repository](https://github.com/githubharald/CTCDecoder) and his [blog](https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7). (Thanks @githubharald)

Data synthesis is based on [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). (Thanks @Belval)

And a good read about CTC from distill.pub [here](https://distill.pub/2017/ctc/).

## Want To Contribute?

Let's advance humanity together by making AI available to everyone!

3 ways to contribute:

**Coder:** Please send a PR for small bugs/improvements. For bigger ones, discuss with us by opening an issue first. There is a list of possible bug/improvement issues tagged with ['PR WELCOME'](https://github.com/JaidedAI/EasyOCR/issues?q=is%3Aissue+is%3Aopen+label%3A%22PR+WELCOME%22).

**User:** Tell us how EasyOCR benefits you/your organization to encourage further development. Also post failure cases in [Issue  Section](https://github.com/JaidedAI/EasyOCR/issues) to help improve future models.

**Tech leader/Guru:** If you found this library useful, please spread the word! (See [Yann Lecun's post](https://www.facebook.com/yann.lecun/posts/10157018122787143) about EasyOCR)

## Guideline for new language request

To request a new language, we need you to send a PR with the 2 following files:

1. In folder [easyocr/character](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character),
we need 'yourlanguagecode_char.txt' that contains list of all characters. Please see format examples from other files in that folder.
2. In folder [easyocr/dict](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict),
we need 'yourlanguagecode.txt' that contains list of words in your language.
On average, we have ~30000 words per language with more than 50000 words for more popular ones.
More is better in this file.

If your language has unique elements (such as 1. Arabic: characters change form when attached to each other + write from right to left 2. Thai: Some characters need to be above the line and some below), please educate us to the best of your ability and/or give useful links. It is important to take care of the detail to achieve a system that really works.

Lastly, please understand that our priority will have to go to popular languages or sets of languages that share large portions of their characters with each other (also tell us if this is the case for your language). It takes us at least a week to develop a new model, so you may have to wait a while for the new model to be released.

See [List of languages in development](https://github.com/JaidedAI/EasyOCR/issues/91)

## Business Inquiries

For Enterprise Support, [Jaided AI](https://www.jaided.ai/) offers full service for custom OCR/AI systems from building, to maintenance and deployment. Click [here](https://www.jaided.ai/contactus) to contact us.
