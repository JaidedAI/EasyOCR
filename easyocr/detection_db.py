'''
Created by Jaided AI
Released Date: 18/08/2022
Description:
A wrapper for DBNet text detection module for EasyOCR
'''
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from .DBNet.DBNet import DBNet

def test_net(image, 
             detector, 
             threshold = 0.2, 
             bbox_min_score = 0.2, 
             bbox_min_size = 3, 
             max_candidates = 0, 
             canvas_size = None, 
             poly = False, 
             device = 'cpu'
             ):
    '''
    A wrapper for DBNet inference routine.

    Parameters
    ----------
    image : np.ndarray or list of np.ndarray
        OpenCV BGR image array or list of it.
    detector : obj
        DBNet text detection object.
    threshold : float, optional
        Minimum probability for each pixel of heatmap tensor to be considered
        as a valid text pixel. The default is 0.2.
    bbox_min_score : float, optional
        Minimum score for each detected bounding box to be considered as a
        valid text bounding box. The default is 0.2.
    bbox_min_size : int, optional
        Minimum size for each detected bounding box to be considered as a
        valid text bounding box. The default is 3.
    max_candidates : int, optional
        Maximum number of detected bounding boxes to be considered as 
        candidates for valid text bounding boxes. Setting to 0 implies
        no maximum. The default is 0.
    canvas_size : int, optional
        Target detection size. Input image will be resized such that it's 
        shorter side is equal to the closest multiple of 32 to the provided 
        canvas_size. If detection_size is not provided, it will be resized to 
        the closest multiple of 32 each side. If the original size exceeds the 
        min-/max-detection sizes (specified in DBNet_inference.yaml), it will be 
        resized to be within the min-/max-sizes. The default is None.
    poly : boolean, optional
        If true, return the bounding boxes as find polygons, otherwise, return
        as rectagular. The default is False.
    device : str, optional
        Device to use. Options are "cpu" and "cuda". The default is 'cpu'.

    Returns
    -------
    bboxes : list of lists
        List of text bounding boxes in format [left, right, top, bottom].
    polys : list of lists
        List of polygon text bounding boxes. If argument poly is set to false,
        this output will also hold the value of output bboxes
    '''
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]
    
    # resize
    images, original_shapes = zip(*[detector.resize_image(img, canvas_size) for img in image_arrs])
    # preprocessing
    images = [np.transpose(detector.normalize_image(n_img), (2, 0, 1)) for n_img in images]
    image_tensor = torch.from_numpy(np.array(images)).to(device)
    # forward pass
    with torch.no_grad():
        hmap = detector.image2hmap(image_tensor.to(device))
        bboxes, _ = detector.hmap2bbox(
                            image_tensor, 
                            original_shapes,
                            hmap, 
                            text_threshold = threshold, 
                            bbox_min_score = bbox_min_score, 
                            bbox_min_size = bbox_min_size, 
                            max_candidates = max_candidates, 
                            as_polygon=False)
        if poly:
            polys, _ = detector.hmap2bbox(
                                image_tensor, 
                                original_shapes,
                                hmap, 
                                text_threshold = threshold, 
                                bbox_min_score = bbox_min_score, 
                                bbox_min_size = bbox_min_size, 
                                max_candidates = max_candidates, 
                                as_polygon=True)
        else:
            polys = bboxes

    return bboxes, polys

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    '''
    A wrapper to initialize DBNet text detection model

    Parameters
    ----------
    trained_model : str
        Path to trained weight to use.
    device : str, optional
        Device to use. Options are "cpu" and "cuda". The default is 'cpu'.
    quantize : boolean, optional
        If use, apply model quantization method to the model. The default is True.
    cudnn_benchmark : boolen, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    dbnet : obj
        DBNet text detection object.
    '''
    dbnet = DBNet(initialize_model = False, 
                  dynamic_import_relative_path = "easyocr/DBNet",
                  device = device, 
                  verbose = 0)
    dbnet.construct_model(dbnet.configs['resnet18']['model'])
    if device == 'cpu':
        dbnet.load_weight(trained_model)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(dbnet, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        dbnet.load_weight(trained_model)
        dbnet.model = torch.nn.DataParallel(dbnet.model).to(device)
        cudnn.benchmark = cudnn_benchmark
    
    dbnet.model.eval()

    return dbnet

def get_textbox(detector, 
                image,
                canvas_size = None, 
                poly = False, 
                threshold = 0.2, 
                bbox_min_score = 0.2, 
                bbox_min_size = 3, 
                max_candidates = 0,
                device = 'cpu',
                **kwargs
                ):
    '''
    A compatibility wrapper to allow supporting calling this method while 
    providing argument for other detector classes and reformat output accordingly.

    Parameters
    ----------
    detector : obj
        DBNet text detection object.
    image : np.ndarray or list of np.ndarray
        OpenCV BGR image array or list of it.
    canvas_size : int, optional
        Target detection size. Please see docstring under method resize_image()
        for explanation. The default is None.
    poly : boolean, optional
        If true, return the bounding boxes as find polygons, otherwise, return
        as rectagular. The default is False.
    threshold : float, optional
        Minimum probability for each pixel of heatmap tensor to be considered
        as a valid text pixel. The default is 0.2.
    bbox_min_score : float, optional
        Minimum score for each detected bounding box to be considered as a
        valid text bounding box. The default is 0.2.
    bbox_min_size : int, optional
        Minimum size for each detected bounding box to be considered as a
        valid text bounding box. The default is 3.
    max_candidates : int, optional
        Maximum number of detected bounding boxes to be considered as 
        candidates for valid text bounding box. Setting it to 0 implies
        no maximum. The default is 0.
    device : str, optional
        Device to use. Options are "cpu" and "cuda". The default is 'cpu'.
    **kwargs : keyword arguments
        Unused. Added to support calling this method while providing argument 
        for other detector class.

    Returns
    -------
    result : list of lists
        List of text bounding boxes in format [left, right, top, bottom].
    '''
    _, polys_list = test_net(image, 
                             detector, 
                             threshold = threshold, 
                             bbox_min_score = bbox_min_score, 
                             bbox_min_size = bbox_min_size, 
                             max_candidates =max_candidates, 
                             canvas_size = canvas_size, 
                             poly = poly, 
                             device = device
                             )
                
    result = [[np.array(box).astype(np.int32).reshape((-1)) for box in polys] for polys in polys_list]

    return result
