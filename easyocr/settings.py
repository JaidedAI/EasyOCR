os.environ["LRU_CACHE_CAPACITY"] = "1"
LOGGER = getLogger(__name__)

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = (
    os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH") or os.path.expanduser("~/.EasyOCR/")
)

# detector parameters
DETECTOR_FILENAME = "craft_mlt_25k.pth"

# recognizer parameters
# fmt: off
latin_lang_list = ['af','az','bs','cs','cy','da','de','en','es','et','fr','ga', \
                   'hr','hu','id','is','it','ku','la','lt','lv','mi','ms','mt', \
                   'nl','no','oc','pl','pt','ro','rs_latin','sk','sl','sq', \
                   'sv','sw','tl','tr','uz','vi']
arabic_lang_list = ['ar','fa','ug','ur']
cyrillic_lang_list = ['ru','rs_cyrillic','be','bg','uk','mn']
devanagari_lang_list = ['hi','mr','ne']
all_lang_list = latin_lang_list + arabic_lang_list+ cyrillic_lang_list + devanagari_lang_list + ['th','ch_sim','ch_tra','ja','ko','ta']
# fmt: on
imgH = 64
input_channel = 1
output_channel = 512
hidden_size = 512

number = "0123456789"
symbol = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
en_char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# first element is url path, second is file size
model_url = {
    "detector": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "2f8227d2def4037cdb3b34389dcf9ec1",
    ),
    "latin.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/latin.zip",
        "fb91b9abf65aeeac95a172291b4a6176",
    ),
    "chinese.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese.zip",
        "dfba8e364cd98ed4fed7ad54d71e3965",
    ),
    "chinese_sim.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese_sim.zip",
        "0e19a9d5902572e5237b04ee29bdb636",
    ),
    "japanese.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/japanese.zip",
        "6d891a4aad9cb7f492809515e4e9fd2e",
    ),
    "korean.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/korean.zip",
        "45b3300e0f04ce4d03dda9913b20c336",
    ),
    "thai.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip",
        "40a06b563a2b3d7897e2d19df20dc709",
    ),
    "devanagari.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip",
        "db6b1f074fae3070f561675db908ac08",
    ),
    "cyrillic.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/cyrillic.zip",
        "5a046f7be2a4f7da6ed50740f487efa8",
    ),
    "arabic.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip",
        "993074555550e4e06a6077d55ff0449a",
    ),
    "tamil.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.1.7/tamil.zip",
        "4b93972fdacdcdabe6d57097025d4dc2",
    ),
}
