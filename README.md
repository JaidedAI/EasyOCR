# Easy OCR

Ready-to-use OCR with 40+ languages supported including Chinese, Japanese, Korean and Thai.

## Examples

![example](examples/example.png)

![example2](examples/example2.png)

## Supported Languages

We are currently supporting following 42 languages.

Afrikaans (af), Azerbaijani (az), Bosnian (bs), Czech (cs), Welsh (cy),
Danish (da), German (de), English (en), Spanish (es), Estonian (et),
French (fr), Irish (ga), Croatian (hr), Hungarian (hu), Indonesian (id),
Icelandic (is), Italian (it), Japanese (ja), Korean (ko), Kurdish (ku),
Latin (la), Lithuanian (lt),
Latvian (lv), Maori (mi), Malay (ms), Maltese (mt), Dutch (nl), Norwegian (no),
Polish (pl), Portuguese (pt),Romanian (ro), Slovak (sk), Slovenian (sl),
Albanian (sq), Swedish (sv),Swahili (sw), Thai (th), Tagalog (tl),
Turkish (tr), Uzbek (uz), Vietnamese (vi), Chinese (zh)

List of characters is in folder easy_ocr/character. If you are native speaker
of any language and think we should add or remove any character,
please create an issue.

## Installation

Install using `pip` for stable release, (to be published soon)

``` bash
pip install easyocr
```

For latest development release,

``` bash
pip install git+git://github.com/jaidedai/easy_ocr.git
```

Note: for Windows, please install torch and torchvision first by following official instruction here https://pytorch.org

## Usage

``` python
import easyocr
reader = easyocr.Reader(['th','en'])
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

There are optional arguments for readtext function, `decoder` can be 'greedy'(default), 'beamsearch', or 'wordbeamsearch'. For 'beamsearch' and 'wordbeamsearch', you can also set `beamWidth` (default=5). Bigger number will be slower but can be more accurate. For multiprocessing, you can set `workers` and `batch_size`. Current version converts image into grey scale for recognition model, so contrast can be an issue. You can try playing with `contrast_ths`, `adjust_contrast` and `filter_ths`.

## To be implemented

1. Language packs: Russian-based languages + Arabic + etc.
2. Language model for better decoding
3. Better documentation and api

## Acknowledgement and References

This project is based on researches/codes from several papers/open-source repositories.

Detection part is using CRAFT algorithm from this [official repository](https://github.com/clovaai/CRAFT-pytorch) and their [paper](https://arxiv.org/abs/1904.01941).

Recognition model is CRNN ([paper](https://arxiv.org/abs/1507.05717)). It is composed of 3 main components, feature extraction (we are currently using [Resnet](https://arxiv.org/abs/1512.03385)), sequence labeling ([LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)) and decoding ([CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). Training pipeline for recognition part is a modified version from this [repository](https://github.com/clovaai/deep-text-recognition-benchmark).

Beam search code is based on this [repository](https://github.com/githubharald/CTCDecoder) and his [blog](https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7).

And good read about CTC from distill.pub [here](https://distill.pub/2017/ctc/).


## Citations

If you use `easy_ocr` in your project/research, please consider citing the library as follows ... (link to be created)
