# Jaided Read

End-to-End Multilingual Optical Character Recognition (OCR) Solution

## Supported Languages

We are currently supporting following 39 languages.

Afrikaans (af), Azerbaijani (az), Bosnian (bs), Czech (cs), Welsh (cy),
Danish (da), German (de), English (en), Spanish (es), Estonian (et),
French (fr), Irish (ga), Croatian (hr), Hungarian (hu), Indonesian (id),
Icelandic (is), Italian (it), Kurdish (ku),  Latin (la), Lithuanian (lt),
Latvian (lv), Maori (mi), Malay (ms), Maltese (mt), Dutch (nl), Norwegian (no),
Polish (pl), Portuguese (pt),Romanian (ro), Slovak (sk), Slovenian (sl),
Albanian (sq), Swedish (sv),Swahili (sw), Thai (th), Tagalog (tl),
Turkish (tr), Uzbek (uz), Vietnamese (vi)

## Installation

Install using `pip` for stable release,

``` bash
pip install jaidedread
```

For latest development release,

``` bash
pip install git+git://github.com/jaided/jaidedread.git
```

## Usage

``` python
import jaidedread
reader = jaidedread.Reader(['th','en'])
reader.readtext('test.jpg')
```

Model weight for chosen language will be automatically downloaded or you can
download it manually from  https://jaided.ai/read_download and put it
in 'model' folder.

Output will be in list format, each item represents bounding box, text and confident level, respectively.

``` bash
[([[1344, 439], [2168, 439], [2168, 580], [1344, 580]], 'ใจเด็ด', 0.4542357623577118),
 ([[1333, 562], [2169, 562], [2169, 709], [1333, 709]], 'Project', 0.9557611346244812)]
```

See full documentation at https://jaided.ai/read/doc

## To be implemented

1. Language packs: Chinese, Japanese, Korean group + Russian-based languages +
Arabic + etc.
2. Language model for better decoding
3. Better documentation and api

## Acknowledgement and References

This project is based on researches/codes from several papers/open-source repositories.

Detection part is using CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941).

Recognition model is CRNN ([paper](https://arxiv.org/abs/1507.05717)). It is composed of 3 main components, feature extraction (we are currently using [Resnet](https://arxiv.org/abs/1512.03385)), sequence labeling ([LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)) and decoding ([CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). Training pipeline for recognition part is a modified version from this [repository](https://github.com/clovaai/deep-text-recognition-benchmark).

Beam search code is based on this [repository](https://github.com/githubharald/CTCDecoder) and his [blog](https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7).

And good read about CTC from distill.pub [here](https://distill.pub/2017/ctc/).


## Citations

For academic research, please cite the library as follows ... (link to be created)
