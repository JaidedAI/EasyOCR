import os
import argparse
import lzma
import pickle
from datetime import datetime
import numpy as np
import PIL.Image

import torch

import easyocr

# %%
def count_parameters(model):
    return sum([param.numel() for param in model.parameters()])

def get_weight_norm(model):
    with torch.no_grad():
        return sum([param.norm() for param in model.parameters()]).cpu().item()
    
def replace(list_in, indices, values):
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(values, list):
        values = [values]
    assert len(indices) == len(values)
    
    list_out = list_in.copy()
    for index, value in zip(indices, values):
        list_out[index] = value

    return list_out

def get_easyocr(language):
    if not isinstance(language, list):
        language = [language]
    return easyocr.Reader(language)

# %%
def main(args):
    
    if args.output is None:
        args.output = "EasyOcrUnitTestPackage_{}.pickle".format(datetime.now().strftime("%Y%m%dT%H%M"))
    
    if args.data_dir is None:
        data_dir = "./examples"
    else:
        data_dir = args.data_dir
    
    image_preprocess = {
        'english.png':{
            "tiny": [540, 420, 690, 470],
            "mini": [260, 90, 605, 160],
            "small": [243, 234, 636, 360]
            }, 
        'french.jpg':{
            "tiny": [184, 615, 425, 732]
            }, 
        'chinese.jpg':{
            "tiny": [181, 78, 469, 157]
            }, 
        'korean.png':{
            "tiny": [130, 84, 285, 180]
            }
        }

    
    if any([file not in os.listdir(data_dir) for file in image_preprocess.keys()]):
        raise FileNotFoundError("Cannot find {} in {}.").format(', '.join([file for file in image_preprocess.keys() if file not in os.listdir(data_dir)], data_dir))
    
    easyocr_config = {"main_language": 'en'}
    
    ocr = get_easyocr(easyocr_config["main_language"])
    
    images = {os.path.splitext(file)[0]: {
                    key: np.asarray(PIL.Image.open(os.path.join(data_dir, file)).crop(crop_box))[:,:,::-1] for (key,crop_box) in page.items() 
                    } for (file,page) in image_preprocess.items()}

    
    english_mini_bgr, english_mini_gray = easyocr.utils.reformat_input(images['english']['mini'])
    english_small_bgr, english_small_gray = easyocr.utils.reformat_input(images['english']['small'])
    
    
    model_init_test = {'test01': {
                            'description': "Counting parameters of detector module.",
                            "method": "unit_test.count_parameters",
                            'input': ["unit_test.easyocr.ocr.detector"],
                            'output': count_parameters(ocr.detector),    
                            'severity': "Error"
                            },
                      'test02': {
                            'description': "Calculating total norm of parameters in detector module.",
                            "method": "unit_test.get_weight_norm",
                            'input': ["unit_test.easyocr.ocr.detector"],
                            'output': get_weight_norm(ocr.detector),    
                            'severity': "Warning"
                            },
                      'test03': {
                            'description': "Counting parameters of recognition module.",
                            "method": "unit_test.count_parameters",
                            'input': ["unit_test.easyocr.ocr.recognizer"],
                            'output': count_parameters(ocr.recognizer),    
                            'severity': "Error"
                            },
                      'test04': {
                            'description': "Calculating total norm of parameters in recognition module.",
                            "method": "unit_test.get_weight_norm",
                            'input': ["unit_test.easyocr.ocr.recognizer"],
                            'output': get_weight_norm(ocr.recognizer),    
                            'severity': "Warning"
                            },
                      }
    
    
    get_textbox_test = {}
    
    input0 = [ocr.detector,#detector
              english_mini_bgr,#image 
              2560,#canvas_size
              1.0,#mag_ratio
              0.7,#text_threshold 
              0.4,#link_threshold 
              0.4,#low_text
              False, #poly #Fixed 
              'cuda', #device #fixed ? 
              ]
    get_textbox_test.update({'test01': {
                                'description': "Testing with default input.",
                                "method": "unit_test.easyocr.detection.get_textbox",
                                'input': replace(input0, 
                                                 [0, 1], 
                                                 ["unit_test.easyocr.ocr.detector",
                                                  "unit_test.inputs.images.english.mini_bgr"
                                                   ]),
                                'output': easyocr.detection.get_textbox(*input0),    
                                'severity': "Error"
                                }})
    
    input0 = [ocr.detector,#detector
              english_mini_bgr,#image 
              1280,#canvas_size
              1.2,#mag_ratio
              0.6,#text_threshold 
              0.3,#link_threshold 
              0.3,#low_text
              False, #poly #Fixed 
              'cuda', #device #fixed ? 
              ]
    
    get_textbox_test.update({'test02': {
                            'description': "Testing with custom input.",
                            "method": "unit_test.easyocr.detection.get_textbox",
                            'input': replace(input0, 
                                             [0, 1], 
                                             ["unit_test.easyocr.ocr.detector",
                                              "unit_test.inputs.images.english.mini_bgr"
                                               ]),
                            'output': easyocr.detection.get_textbox(*input0),
                            'severity': "Error"
                            }})
    
    input0 = [ocr.detector,#detector
              english_mini_bgr,#image 
              640,#canvas_size
              0.8,#mag_ratio
              0.8,#text_threshold 
              0.5,#link_threshold 
              0.5,#low_text
              False, #poly #Fixed 
              'cuda', #device #fixed ? 
              ]
    
    get_textbox_test.update({'test03': {
                            'description': "Testing with custom input.",
                            "method": "unit_test.easyocr.detection.get_textbox",
                            'input': replace(input0, 
                                             [0, 1], 
                                             ["unit_test.easyocr.ocr.detector",
                                              "unit_test.inputs.images.english.mini_bgr"
                                               ]),
                            'output': easyocr.detection.get_textbox(*input0),
                            'severity': "Error"
                            }})
    

    input0 = [ocr.detector,#detector
              english_mini_bgr,#image 
              2560,#canvas_size
              1.0,#mag_ratio
              0.7,#text_threshold 
              0.4,#link_threshold 
              0.4,#low_text
              False, #poly #Fixed 
              'cuda', #device #fixed ? 
              ]
    output0 = easyocr.detection.get_textbox(*input0)
    polys = output0[0]
    group_text_box_test = {}
    
    input_ = [polys, 
              0.1,# slope_ths 
              0.5,#ycenter_ths
              0.5,#height_ths
              1.0,#width_ths 
              0.05,#add_margin 
              True#sort_output
              ]
    group_text_box_test.update({'test01': {
                                'description': "Testing with default input.",
                                "method": "unit_test.easyocr.utils.group_text_box",
                                'input': input_,
                                'output': easyocr.utils.group_text_box(*input_),    
                                'severity': "Error"
                                }
                            })
    input_ = [polys, 
              0.05,# slope_ths 
              0.3,#ycenter_ths
              0.3,#height_ths
              0.8,#width_ths 
              0.03,#add_margin 
              True#sort_output
              ]
    group_text_box_test.update({'test02': {
                                'description': "Testing with custom input.",
                                "method": "unit_test.easyocr.utils.group_text_box",
                                'input': input_,
                                'output': easyocr.utils.group_text_box(*input_),    
                                'severity': "Error"
                                }
                            })
    input_ = [polys, 
              0.12,# slope_ths 
              0.7,#ycenter_ths
              0.7,#height_ths
              1.2,#width_ths 
              0.1,#add_margin 
              True#sort_output
              ]
    group_text_box_test.update({'test03': {
                                'description': "Testing with custom input.",
                                "method": "unit_test.easyocr.utils.group_text_box",
                                'input': input_,
                                'output': easyocr.utils.group_text_box(*input_),    
                                'severity': "Error"
                                }
                            })
    
    input0 = [None, 
              20, #min_size
              0.7, #text_threshold - fixed
              0.4, #low_text - fixed
              0.4, # link_threshold - fixed
              2560, #canvas_size -fixed
              1., #mag_ratio - fixed
              0.1, #slope_ths - fixed
              0.5, #ycenter_ths - fixed
              0.5, #height_ths - fixed
              0.5, #width_ths - fixed
              0.1, #add_margin - fixed
              True, #reformat - fixed
              None #optimal_num_chars  - fixed
              ]
    
    detect_test = {}

    input_ = replace(input0, [0,1], [english_mini_bgr, 20])
    detect_test.update({'test01': {
                        'description': "Testing with default input.",
                        "method": "unit_test.easyocr.ocr.detect",
                        'input': replace(input_, 0, "unit_test.inputs.images.english.mini_bgr"),
                        'output': ocr.detect(*input_),    
                        'severity': "Error"
                        },
                    })
    input_ = replace(input0, [0,1], [english_small_bgr, 20])
    detect_test.update({'test02': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.ocr.detect",
                        'input': replace(input_, 0, "unit_test.inputs.images.english.small_bgr"),
                        'output': ocr.detect(*input_),    
                        'severity': "Error"
                        },
                    })
    input_ = replace(input0, [0,1], [english_small_bgr, 100])
    detect_test.update({'test03': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.ocr.detect",
                        'input': replace(input_, 0, "unit_test.inputs.images.english.small_bgr"),
                        'output': ocr.detect(*input_),    
                        'severity': "Error"
                        },
                    })
    
    get_image_list_test = {}
    output0 = ocr.detect(english_small_bgr)
    input0 = [output0[0][0], 
              output0[1][0], 
              english_small_gray, 
              64, #model_height 
              True# sort_output
              ]
    input_ = replace(input0, 2, "unit_test.inputs.images.english.small_gray")
    get_image_list_test.update({'test01': {
                        'description': "Testing with default input.",
                        "method": "unit_test.easyocr.utils.get_image_list",
                        'input': input_,
                        'output': easyocr.utils.get_image_list(*input0),    
                        'severity': "Error"
                        },
                    })
    
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], 
              output0[1][0], 
              english_mini_gray, 
              64, #model_height 
              True# sort_output
              ]
    input_ = replace(input0, 2, "unit_test.inputs.images.english.mini_gray")
    get_image_list_test.update({'test02': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.utils.get_image_list",
                        'input': input_,
                        'output': easyocr.utils.get_image_list(*input0),    
                        'severity': "Error"
                        },
                    })
    
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], 
              output0[1][0], 
              english_mini_gray, 
              64, #model_height 
              True# sort_output
              ]
    image_list, max_width = easyocr.utils.get_image_list(*input0)
    
    input0 = [ocr.character, 
              64, #imgH - fixed 
              int(max_width), 
              ocr.recognizer, 
              ocr.converter, 
              image_list[:2],
              '', #ignore_char, 
              'greedy', #decoder, 
              5, #beamWidth, 
              1, #batch_size, 
              0.1, #contrast_ths, 
              0.5, #adjust_contrast, 
              0.003, #filter_ths,
              1, #workers, 
              "cuda" #device
              ]
    
    get_text_test = {}
        
    output_ = easyocr.recognition.get_text(*input0)   
    input_ = replace(input0, 
                     [0, 3, 4], 
                     ["unit_test.easyocr.ocr.character", 
                      "unit_test.easyocr.ocr.recognizer", 
                      "unit_test.easyocr.ocr.converter"]
                     )

    get_text_test.update({'test01': {
                        'description': "Testing with default input.",
                        "method": "unit_test.easyocr.recognition.get_text",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        },
                    })
    
    input0 = [ocr.character, 
              64, #imgH - fixed 
              int(max_width), 
              ocr.recognizer, 
              ocr.converter, 
              image_list[:2],
              '', #ignore_char, 
              'greedy', #decoder, 
              4, #beamWidth, 
              1, #batch_size, 
              0.05, #contrast_ths, 
              0.3, #adjust_contrast, 
              0.001, #filter_ths,
              1, #workers, 
              "cuda" #device
              ]
    
    output_ = easyocr.recognition.get_text(*input0)   
    input_ = replace(input0, 
                     [0, 3, 4], 
                     ["unit_test.easyocr.ocr.character", 
                      "unit_test.easyocr.ocr.recognizer", 
                      "unit_test.easyocr.ocr.converter"]
                     )
    get_text_test.update({'test02': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.recognition.get_text",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    
    input0 = [ocr.character, 
              64, #imgH - fixed 
              int(max_width), 
              ocr.recognizer, 
              ocr.converter, 
              image_list[:2],\
              '', #ignore_char, 
              'greedy', #decoder, 
              6, #beamWidth, 
              4, #batch_size, 
              0.2, #contrast_ths, 
              0.6, #adjust_contrast, 
              0.005, #filter_ths,
              1, #workers, 
              "cuda" #device
              ]
    
    output_ = easyocr.recognition.get_text(*input0)   
    input_ = replace(input0, 
                     [0, 3, 4], 
                     ["unit_test.easyocr.ocr.character", 
                      "unit_test.easyocr.ocr.recognizer", 
                      "unit_test.easyocr.ocr.converter"]
                     )
    get_text_test.update({'test03': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.recognition.get_text",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    
    
    get_paragraph_test = {}
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], 
              output0[1][0], 
              english_mini_gray, 
              64, #model_height 
              True# sort_output
              ]
    image_list, max_width = easyocr.utils.get_image_list(*input0)
    
    input0 = [ocr.character, 
              64, #imgH - fixed 
              int(max_width), 
              ocr.recognizer, 
              ocr.converter, 
              image_list[:2],
              '', #ignore_char, 
              'greedy', #decoder, 
              5, #beamWidth, 
              1, #batch_size, 
              0.1, #contrast_ths, 
              0.5, #adjust_contrast, 
              0.003, #filter_ths,
              1, #workers, 
              "cuda" #device
              ]
    
    output0 = easyocr.recognition.get_text(*input0)   
    input_ = [output0, 
              1, #x_ths
              0.5, #y_ths 
              'ltr' #mode
              ]
    get_paragraph_test.update({'test01': {
                        'description': "Testing with default input.",
                        "method": "unit_test.easyocr.utils.get_paragraph",
                        'input': input_,
                        'output': easyocr.utils.get_paragraph(*input_),    
                        'severity': "Error"
                        }})
    input_ = [output0, 
              0.5, #x_ths
              0.3, #y_ths 
              'ltr' #mode
              ]
    get_paragraph_test.update({'test02': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.utils.get_paragraph",
                        'input': input_,
                        'output': easyocr.utils.get_paragraph(*input_),    
                        'severity': "Error"
                        }})
    input_ = [output0, 
              1.5, #x_ths
              1, #y_ths 
              'ltr' #mode
              ]
    get_paragraph_test.update({'test03': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.utils.get_paragraph",
                        'input': input_,
                        'output': easyocr.utils.get_paragraph(*input_),    
                        'severity': "Error"
                        }})
    
    
    input_recog = [None, 
              None, #horizontal_list
              None, #free_list
              'greedy', #decoder
              5, #beamWidth
              1,#batch_size
              0, #workers
              None, #allowlist
              None, #blocklist
              1, #detail
              None, #rotation_info
              False,#paragraph
              0.1,#contrast_ths
              0.5, #adjust_contrast
              0.003, #filter_ths
              0.5, #y_ths
              1.0, #x_ths
              True, #reformat
              'standard'#output_format
              ]
    
    recognize_test = {}
    
    h_list, f_list = ocr.detect(english_mini_bgr)
    input_ = replace(input_recog, 
                     [0, 1, 2], 
                     [english_mini_gray, h_list[0], f_list[0]])
    recognize_test.update({'test01': {
                        'description': "Testing with default input.",
                        "method": "unit_test.easyocr.ocr.recognize",
                        'input': replace(input_, 0, "unit_test.inputs.images.english.mini_gray"),
                        'output': ocr.recognize(*input_),    
                        'severity': "Error"
                        }})
    
    h_list, f_list = ocr.detect(english_small_bgr)
    input_ = replace(input_recog, 
                     [0, 1, 2], 
                     [english_small_gray, h_list[0], f_list[0]])
    recognize_test.update({'test02': {
                        'description': "Testing with custom input.",
                        "method": "unit_test.easyocr.ocr.recognize",
                        'input': replace(input_, 0, "unit_test.inputs.images.english.small_gray"),
                        'output': ocr.recognize(*input_),    
                        'severity': "Error"
                        }})

    readtext_test = {}
    #english_tiny_bgr, _ = easyocr.utils.reformat_input(images['english']['tiny'])
    input_ = ["unit_test.inputs.images.english.tiny", 'en']
    ocr = get_easyocr('en')
    _, pred, confidence = ocr.readtext(images['english']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test01': {
                        'description': "Reading English text.",
                        "method": "unit_test.easyocr_read_as",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    #french_tiny_bgr, _ = easyocr.utils.reformat_input(images['french']['tiny'])
    input_ = ["unit_test.inputs.images.french.tiny", 'fr']
    ocr = get_easyocr('fr')
    _, pred, confidence = ocr.readtext(images['french']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test02': {
                        'description': "Reading French text.",
                        "method": "unit_test.easyocr_read_as",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    #chinese_tiny_bgr, _ = easyocr.utils.reformat_input(images['chinese']['tiny'])
    input_ = ["unit_test.inputs.images.chinese.tiny", 'ch_sim']
    ocr = get_easyocr('ch_sim')
    _, pred, confidence = ocr.readtext(images['chinese']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test03': {
                        'description': "Reading Chinese (simplified) text.",
                        "method": "unit_test.easyocr_read_as",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    #korean_tiny_bgr, _ = easyocr.utils.reformat_input(images['korean']['tiny'])
    input_ = ["unit_test.inputs.images.korean.tiny", 'ko']
    ocr = get_easyocr('ko')
    _, pred, confidence = ocr.readtext(images['korean']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test04': {
                        'description': "Reading Korean text.",
                        "method": "unit_test.easyocr_read_as",
                        'input': input_,
                        'output': output_,    
                        'severity': "Error"
                        }})
    
    
    
    solution_book = {
            'inputs':{'images': image_preprocess,
                      'easyocr_config': easyocr_config
                      },
            'tests':{
                 "model initialization": model_init_test,
                 "get_textbox function": get_textbox_test,
                 "group_text_box function": group_text_box_test,
                 "detect method": detect_test,
                 "get_image_list function": get_image_list_test,
                 "get_text_test function": get_text_test,
                 "get_paragraph_test function": get_paragraph_test,
                 "recognize method": recognize_test,
                 "readtext method": readtext_test,
                 }
            }
            
    
    
    with lzma.open(args.output, 'wb') as fid:
        pickle.dump(solution_book, fid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to pack EasyOCR weight.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", default=None, help="output path.")
    parser.add_argument("-d", "--data_dir", default=None, help="data directory")
    args = parser.parse_args()
    main(args)














