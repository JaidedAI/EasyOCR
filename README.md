# EasyOCR

[![PyPI Status](https://badge.fury.io/py/easyocr.svg)](https://badge.fury.io/py/easyocr)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.fan/easyocr)
[![Tweet](https://img.shields.io/twitter/url/https/github.com/JaidedAI/EasyOCR.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20library:%20EasyOCR%20https://github.com/JaidedAI/EasyOCR)
[![Twitter](https://img.shields.io/badge/twitter-@JaidedAI-blue.svg?style=flat)](https://twitter.com/JaidedAI)

Ready-to-use OCR with 80+ languages supported including Chinese, Japanese, Korean and Thai.

## What's new
- 7 February 2021 - Version 1.2.4
    - Faster CPU inference speed by using dynamic input shape (recognition rate increases by around 100% for images with a lot of text)
- 1 February 2021 - Version 1.2.3
    - Add `setLanguageList` method to `Reader` class. This is a convenient api for changing languages (within the same model) after creating class instance.
    - Small change on text box merging. (thanks [z-pc](https://github.com/z-pc), see [PR](https://github.com/JaidedAI/EasyOCR/pull/338))
    - [Basic Demo on website](https://www.jaided.ai/easyocr)
- 5 January 2021 - Version 1.2.2
    - Add `optimal_num_chars` to `detect` method. If specified, bounding boxes with estimated number of characters near this value are returned first. (thanks [@adamfrees](https://github.com/adamfrees))
    - Add `rotation_info` to `readtext` method. Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations. (thanks [@mijoo308](https://github.com/mijoo308))
    - Update [documentation](https://www.jaided.ai/easyocr/documentation).

- [Read all released notes](https://github.com/JaidedAI/EasyOCR/blob/master/releasenotes.md)

## What's coming next
- Faster processing time
- [New language support](https://github.com/JaidedAI/EasyOCR/issues/91)

## Examples

![example](examples/example.png)

![example2](examples/example2.png)

![example3](examples/example3.png)

## Supported Languages

We are currently supporting 80+ languages. See [list of supported languages](https://www.jaided.ai/easyocr).

## Demo

[Try our demo on website: Jaided AI](https://www.jaided.ai/easyocr)

## Installation

Install using `pip` for stable release,

``` bash
pip install easyocr
```

For latest development release,

``` bash
pip install git+git://github.com/jaidedai/easyocr.git
```

Note 1: for Windows, please install torch and torchvision first by following the official instruction here https://pytorch.org. On pytorch website, be sure to select the right CUDA version you have. If you intend to run on CPU mode only, select CUDA = None.

Note 2: We also provide Dockerfile [here](https://github.com/JaidedAI/EasyOCR/blob/master/Dockerfile).

## Usage

``` python
import easyocr
reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
result = reader.readtext('chinese.jpg')
```

Output will be in list format, each item represents bounding box, text and confident level, respectively.

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
English is compatible with every languages. Languages that share common characters are usually compatible with each other.

Note 2: Instead of filepath `chinese.jpg`, you can also pass OpenCV image object (numpy array) or image file as bytes. URL to raw image is also acceptable.

Note 3: The line `reader = easyocr.Reader(['ch_sim','en'])` is for loading model into memory. It takes some time but it need to be run only once.

You can also set `detail` = 0 for simpler output.

``` python
reader.readtext('chinese.jpg', detail = 0)
```
Result:
``` bash
['愚园路', '西', '东', '315', '309', 'Yuyuan Rd.', 'W', 'E']
```

Model weight for chosen language will be automatically downloaded or you can
download it manually from the following links and put it in '~/.EasyOCR/model' folder

- [text detection model (CRAFT)](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip)
- [latin model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/latin.zip)
- [chinese (traditional) model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese.zip)
- [chinese (simplified) model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese_sim.zip)
- [japanese model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/japanese.zip)
- [korean model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/korean.zip)
- [thai model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip)
- [devanagari model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip)
- [cyrillic model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/cyrillic.zip)
- [arabic model](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip)
- [tamil model](https://github.com/JaidedAI/EasyOCR/releases/download/v1.1.7/tamil.zip)
- [bengali model](https://github.com/JaidedAI/EasyOCR/releases/download/v1.1.8/bengali.zip)
- [telugu model](https://github.com/JaidedAI/EasyOCR/releases/download/v1.2/telugu.zip)
- [kannada model](https://github.com/JaidedAI/EasyOCR/releases/download/v1.2/kannada.zip)

In case you do not have GPU or your GPU has low memory, you can run it in CPU mode by adding gpu = False

``` python
reader = easyocr.Reader(['ch_sim','en'], gpu = False)
```

For more information, read [tutorial](https://www.jaided.ai/easyocr/tutorial) and [API Documentation](https://www.jaided.ai/easyocr/documentation).

#### Run on command line

```shell
$ easyocr -l ch_sim en -f chinese.jpg --detail=1 --gpu=True
```

## Implementation Roadmap

1. Language packs: Expand support to more languages. We are aiming to cover > 80-90% of world's population. Also improve existing languages.
2. Better documentation and api
3. Language model for better decoding
4. Handwritten support: The key is using GAN to generate realistic handwritten dataset.
5. Faster processing time: model pruning (lite version) / quantization / export to other platforms (ONNX?)
6. Open Dataset and model training pipeline
7. Restructure code to support swappable detection and recognition algorithm.
The api should be as easy as
``` python
reader = easyocr.Reader(['en'], detection='DB', recognition = 'CNN_Transformer')
```
The idea is to be able to plug-in any state-of-the-art model into EasyOCR. There are a lot of geniuses trying to make better detection/recognition model. We are not trying to be a genius here, just make genius's works quickly accessible to the public ... for free. (well I believe most geniuses want their work to create positive impact as fast/big as possible) The pipeline should be something like below diagram. Grey slots are placeholders for changeable light blue modules.

![plan](examples/easyocr_framework.jpeg)

## Acknowledgement and References

This project is based on researches/codes from several papers/open-source repositories.

All deep learning part is based on [Pytorch](https://pytorch.org). :heart:

Detection part is using CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941) (Thanks @YoungminBaek from @clovaai). We also use their pretrained model.

Recognition model is CRNN ([paper](https://arxiv.org/abs/1507.05717)). It is composed of 3 main components, feature extraction (we are currently using [Resnet](https://arxiv.org/abs/1512.03385)), sequence labeling ([LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)) and decoding ([CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). Training pipeline for recognition part is a modified version from  [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). (Thanks @ku21fan from @clovaai) This repository is a gem that deserved more recognition.

Beam search code is based on this [repository](https://github.com/githubharald/CTCDecoder) and his [blog](https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7). (Thanks @githubharald)

Data synthesis is based on [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). (Thanks @Belval)

And good read about CTC from distill.pub [here](https://distill.pub/2017/ctc/).

## Want To Contribute?

Let's advance humanity together by making AI available to everyone!

3 ways to contribute:

**Coder:** Please send PR for small bug/improvement. For bigger one, discuss with us by open an issue first. There is a list of possible bug/improvement issue tagged with ['PR WELCOME'](https://github.com/JaidedAI/EasyOCR/issues?q=is%3Aissue+is%3Aopen+label%3A%22PR+WELCOME%22).

**User:** Tell us how EasyOCR benefit you/your organization to encourage further development. Also post failure cases in [Issue  Section](https://github.com/JaidedAI/EasyOCR/issues) to help improving future model.

**Tech leader/Guru:** If you found this library useful, please spread the word! (See [Yann Lecun's post](https://www.facebook.com/yann.lecun/posts/10157018122787143) about EasyOCR)

## Guideline for new language request

To request a new language support, I need you to send a PR with 2 following files

1. In folder [easyocr/character](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character),
we need 'yourlanguagecode_char.txt' that contains list of all characters. Please see format example from other files in that folder.
2. In folder [easyocr/dict](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict),
we need 'yourlanguagecode.txt' that contains list of words in your language.
On average we have ~30000 words per language with more than 50000 words for popular one.
More is better in this file.

If your language has unique elements (such as 1. Arabic: characters change form when attach to each other + write from right to left 2. Thai: Some characters need to be above the line and some below), please educate me with your best ability and/or give useful links. It is important to take care of the detail to achieve a system that really works.

Lastly, please understand that my priority will have to go to popular language or set of languages that share most of characters together (also tell me if your language share a lot of characters with other). It takes me at least a week to work for new model. You may have to wait a while for new model to be released.

See [List of languages in development](https://github.com/JaidedAI/EasyOCR/issues/91)

## Business Inquiries

For Enterprise Support, [Jaided AI](https://www.jaided.ai/) offers full service for custom OCR/AI systems from building, maintenance and deployment. Click [here](https://www.jaided.ai/contact) to contact us.
