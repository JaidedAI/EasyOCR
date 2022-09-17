# Custom recognition models

## How to train your custom model

You can use your own data or generate your own dataset. To generate your own data, we recommend using
[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). We provide an example of a dataset [here](https://jaided.ai/easyocr/modelhub/).
After you have a dataset, you can train your own model by following this repository
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
The network needs to be fully convolutional in order to predict flexible text length. Our current network is 'None-VGG-BiLSTM-CTC'.
Once you have your trained model (a `.pth` file), you need 2 additional files describing recognition network architecture and model configuration.
An example is provided in `custom_example.zip` file [here](https://jaided.ai/easyocr/modelhub/).

Please do not create an issue about data generation and model training in this repository. If you have any question regarding data generation and model training, please ask in the respective repositories.

Note: We also provide our version of a training script [here](https://github.com/JaidedAI/EasyOCR/tree/master/trainer). It is a modified version from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

## How to use your custom model

To use your own recognition model, you need the three files as explained above. These three files have to share the same name (i.e. `yourmodel.pth`, `yourmodel.yaml`, `yourmodel.py`) that you will then use to call your model with EasyOCR API.

We provide [custom_example.zip](https://jaided.ai/easyocr/modelhub/)
as an example. Please download, extract and place `custom_example.py`, `custom_example.yaml` in the `user_network_directory` (default = `~/.EasyOCR/user_network`) and place `custom_example.pth` in model directory (default = `~/.EasyOCR/model`)
Once you place all 3 files in their respective places, you can use `custom_example` by
specifying `recog_network` like this `reader = easyocr.Reader(['en'], recog_network='custom_example')`.
