'''
Created by Jaided AI
Released Date: 18/08/2022
Description:
DBNet text detection module. 
Many parts of the codes are adapted from https://github.com/MhLiao/DB
'''
import os
import math
import yaml
from shapely.geometry import Polygon
import PIL.Image
import numpy as np
import cv2
import pyclipper
import torch

from .model.constructor import Configurable
# %%
class DBNet:
    def __init__(self, 
                 backbone = "resnet18",
                 weight_dir = "./DBNet/weights/",
                 weight_name = 'pretrained',
                 initialize_model = True,
                 dynamic_import_relative_path = None,
                 device = 'cuda', 
                 verbose = 0):
        '''
        DBNet text detector class

        Parameters
        ----------
        backbone : str, optional
            Backbone to use. Options are "resnet18" and "resnet50". The default is "resnet18".
        weight_dir : str, optional
            Path to directory that contains weight files. The default is "./DBNet/weights/".
        weight_name : str, optional
            Name of the weight to use as specified in DBNet_inference.yaml or a filename 
            in weight_dir. The default is 'pretrained'.
        initialize_model : Boolean, optional
            If True, construct the model and load weight at class initialization.
            Otherwise, only initial the class without constructing the model.
            The default is True.
        dynamic_import_relative_path : str, optional
            Relative path to 'model/detector.py'. This option is for supporting
            integrating this module into other modules. For example, easyocr/DBNet
            This should be left as None when calling this module as a standalone. 
            The default is None.
        device : str, optional
            Device to use. Options are "cuda" and "cpu". The default is 'cuda'.
        verbose : int, optional
            Verbosity level. The default is 0.

        Raises
        ------
        ValueError
            Raised when backbone is invalid.
        FileNotFoundError
            Raised when weight file is not found.

        Returns
        -------
        None.
        '''
        self.device = device
        
        config_path = os.path.join(os.path.dirname(__file__), "configs", "DBNet_inference.yaml")
        with open(config_path, 'r') as fid:
            self.configs = yaml.safe_load(fid)

        if dynamic_import_relative_path is not None:
            self.configs = self.set_relative_import_path(self.configs, dynamic_import_relative_path)

        if backbone in self.configs.keys():
            self.backbone = backbone
        else:
            raise ValueError("Invalid backbone. Current support backbone are {}.".format(",".join(self.configs.keys())))

        if initialize_model:
            if weight_name in self.configs[backbone]['weight'].keys():
                weight_path = os.path.join(os.path.dirname(__file__), 'weights', self.configs[backbone]['weight'][weight_name])
                error_message = "A weight with a name {} is found in DBNet_inference.yaml but cannot be find file: {}."
            else:
                weight_path = os.path.join(os.path.dirname(__file__), 'weights', weight_name)
                error_message = "A weight with a name {} is not found in DBNet_inference.yaml and cannot be find file: {}."
                
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(error_message.format(weight_name, weight_path))
                
            self.initialize_model(self.configs[backbone]['model'], weight_path)
        
        else:
            self.model = None

        self.BGR_MEAN = np.array(self.configs['BGR_MEAN'])
        self.min_detection_size = self.configs['min_detection_size']
        self.max_detection_size = self.configs['max_detection_size']

    def set_relative_import_path(self, configs, dynamic_import_relative_path):
        '''
        Create relative import paths for modules specified in class. This method
        is recursive.

        Parameters
        ----------
        configs : dict
            Configuration dictionary from .yaml file.
        dynamic_import_relative_path : str, optional
            Relative path to 'model/detector/'. This option is for supporting
            integrating this module into other modules. For example, easyocr/DBNet
            This should be left as None when calling this module as a standalone. 
            The default is None.
        
        Returns
        -------
        configs : dict
            Configuration dictionary with correct relative path.
        '''
        assert dynamic_import_relative_path is not None
        prefices = dynamic_import_relative_path.split(os.sep)
        for key,value in configs.items():
            if key == 'class':
                configs.update({key: ".".join(prefices + value.split("."))})
            else:
                if isinstance(value, dict):
                    value = self.set_relative_import_path(value, dynamic_import_relative_path)
                else:
                    pass
        return configs

    def load_weight(self, weight_path):
        '''
        Load weight to model.

        Parameters
        ----------
        weight_path : str
            Path to trained weight.

        Raises
        ------
        RuntimeError
            Raised when the model has not yet been contructed.

        Returns
        -------
        None.
        '''
        if self.model is None:
            raise RuntimeError("model has not yet been constructed.")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False)
        self.model.eval()

    def construct_model(self, config):
        '''
        Contruct text detection model based on the configuration in .yaml file.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        None.
        '''
        self.model = Configurable.construct_class_from_config(config).structure.builder.build(self.device)

    def initialize_model(self, model_config, weight_path):
        '''
        Wrapper to initialize text detection model. This model includes contructing
        and weight loading.

        Parameters
        ----------
        model_config : dict
            Configuration dictionary.
        weight_path : str
            Path to trained weight.

        Returns
        -------
        None.
        '''
        self.construct_model(model_config)
        self.load_weight(weight_path)
        
    def get_cv2_image(self, image):
        '''
        Load or convert input to OpenCV BGR image numpy array.

        Parameters
        ----------
        image : str, PIL.Image, or np.ndarray
            Image to load or convert.

        Raises
        ------
        FileNotFoundError
            Raised when the input is a path to file (str), but the file is not found.
        TypeError
            Raised when the data type of the input is not supported.

        Returns
        -------
        image : np.ndarray
            OpenCV BGR image.
        '''
        if isinstance(image, str):
            if os.path.isfile(image):
                image = cv2.imread(image, cv2.IMREAD_COLOR).astype('float32')
            else:
                raise FileNotFoundError("Cannot find {}".format(image))
        elif isinstance(image, np.ndarray):
            image = image.astype('float32')
        elif isinstance(image, PIL.Image.Image):
            image = np.asarray(image)[:, :, ::-1]
        else:
            raise TypeError("Unsupport image format. Only path-to-file, opencv BGR image, and PIL image are supported.")

        return image

    def resize_image(self, img, detection_size = None):
        '''
        Resize image such that the shorter side of the image is equal to the 
        closest multiple of 32 to the provided detection_size. If detection_size
        is not provided, it will be resized to the closest multiple of 32 each
        side. If the original size exceeds the min-/max-detection sizes 
        (specified in configs.yaml), it will be resized to be within the 
        min-/max-sizes.

        Parameters
        ----------
        img : np.ndarray
            OpenCV BGR image.
        detection_size : int, optional
            Target detection size. The default is None.

        Returns
        -------
        np.ndarray
            Resized OpenCV BGR image. The width and height of this image should
            be multiple of 32.
        '''
        height, width, _ = img.shape
        if detection_size is None:
            detection_size = max(self.min_detection_size, min(height, width, self.max_detection_size))
        
        if height < width:
            new_height = int(math.ceil(detection_size / 32) * 32)
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = int(math.ceil(detection_size / 32) * 32)
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))

        return resized_img, (height, width)

    def image_array2tensor(self, image):
        '''
        Convert image array (assuming OpenCV BGR format) to image tensor.

        Parameters
        ----------
        image : np.ndarray
            OpenCV BGR image.

        Returns
        -------
        torch.tensor
            Tensor image with 4 dimension [batch, channel, width, height].
        '''
        return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    def normalize_image(self, image):
        '''
        Normalize image by substracting BGR mean and divided by 255

        Parameters
        ----------
        image : np.ndarray
            OpenCV BGR image.

        Returns
        -------
        np.ndarray
            OpenCV BGR image.
        '''
        return (image - self.BGR_MEAN)/255.0    
       
    def load_image(self, image_path, detection_size = 0):
        '''
        Wrapper to load and convert an image to an image tensor

        Parameters
        ----------
        image : path-to-file, PIL.Image, or np.ndarray
            Image to load or convert.
        detection_size : int, optional
            Target detection size. The default is None.

        Returns
        -------
        img : torch.tensor
            Tensor image with 4 dimension [batch, channel, width, height]..
        original_shape : tuple
            A tuple (height, width) of the original input image before resizing.
        '''
        img =self.get_cv2_image(image_path)
        img, original_shape = self.resize_image(img, detection_size = detection_size)
        img = self.normalize_image(img)
        img = self.image_array2tensor(img)

        return img, original_shape
    
    def load_images(self, images, detection_size = None):
        '''
        Wrapper to load or convert list of multiple images to a single image 
        tensor. Multiple images are concatenated together on the first dimension.
        
        Parameters
        ----------
        images : a list of path-to-file, PIL.Image, or np.ndarray
            Image to load or convert.
        detection_size : int, optional
            Target detection size. The default is None.

        Returns
        -------
        img : torch.tensor
            A single tensor image with 4 dimension [batch, channel, width, height].
        original_shape : tuple
            A list of tuples (height, width) of the original input image before resizing.
        '''
        images, original_shapes = zip(*[self.load_image(image, detection_size = detection_size) 
                                        for image in images])
        return torch.cat(images, dim = 0), original_shapes
    
    def hmap2bbox(self, 
                  image_tensor, 
                  original_shapes,
                  hmap, 
                  text_threshold = 0.2, 
                  bbox_min_score = 0.2, 
                  bbox_min_size = 3, 
                  max_candidates = 0, 
                  as_polygon=False):
        '''
        Translate probability heatmap tensor to text region boudning boxes.

        Parameters
        ----------
        image_tensor : torch.tensor
            Image tensor.
        original_shapes : tuple
            Original size of the image (height, width) of the input image (before
            rounded to the closest multiple of 32).
        hmap : torch.tensor
            Probability heatmap tensor.
        text_threshold : float, optional
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
        as_polygon : boolean, optional
            If True, return the bounding box as polygon (fine vertrices), 
            otherwise return as rectangular. The default is False.

        Returns
        -------
        boxes_batch : list of lists
            Bounding boxes of each text box.
        scores_batch : list of floats
            Confidence scores of each text box.

        '''
        segmentation = self.binarize(hmap, threshold = text_threshold)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(image_tensor.size(0)):
            height, width = original_shapes[batch_index]
            if as_polygon:
                boxes, scores = self.polygons_from_bitmap(
                                        hmap[batch_index],
                                        segmentation[batch_index], 
                                        width, 
                                        height, 
                                        bbox_min_score = bbox_min_score, 
                                        bbox_min_size = bbox_min_size, 
                                        max_candidates = max_candidates)
            else:
                boxes, scores = self.boxes_from_bitmap(
                                        hmap[batch_index],
                                        segmentation[batch_index], 
                                        width, 
                                        height, 
                                        bbox_min_score = bbox_min_score, 
                                        bbox_min_size = bbox_min_size, 
                                        max_candidates = max_candidates)

            boxes_batch.append(boxes)
            scores_batch.append(scores)
            
        boxes_batch, scores_batch = zip(*[zip(*[(box, score) 
                                                for (box,score) in zip(boxes, scores) if score > 0]) 
                                     for (boxes, scores) in zip(boxes_batch, scores_batch)]
                                     )
            
        return boxes_batch, scores_batch
    
    def binarize(self, tensor, threshold):
        '''
        Apply threshold to return boolean tensor.

        Parameters
        ----------
        tensor : torch.tensor
            input tensor.
        threshold : float
            Threshold.

        Returns
        -------
        torch.tensor
            Boolean tensor.

        '''
        return tensor > threshold
    
    def polygons_from_bitmap(self, 
                             hmap,
                             segmentation,
                             dest_width, 
                             dest_height, 
                             bbox_min_score = 0.2, 
                             bbox_min_size = 3, 
                             max_candidates = 0):
        '''
        Translate boolean tensor to fine polygon indicating text bounding boxes

        Parameters
        ----------
        hmap : torch.tensor
            Probability heatmap tensor.
        segmentation : torch.tensor
            Segmentataion tensor.
        dest_width : TYPE
            target width of the output.
        dest_height : TYPE
            target width of the output.
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
        
        Returns
        -------
        boxes_batch : list of lists
            Polygon bounding boxes of each text box.
        scores_batch : list of floats
            Confidence scores of each text box.

        '''
        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]  # The first channel
        hmap = hmap.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        boxes = []
        scores = []
    
        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if max_candidates > 0:
            contours = contours[:max_candidates]
        
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(hmap, points.reshape(-1, 2))
            if score < bbox_min_score:
                continue
            
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue

            else:
                continue

            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < bbox_min_size + 2:
                continue
    
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)

        return boxes, scores
    
    def boxes_from_bitmap(self, 
                          hmap,
                          segmentation,
                          dest_width, 
                          dest_height, 
                          bbox_min_score = 0.2, 
                          bbox_min_size = 3, 
                          max_candidates = 0):
        '''
        Translate boolean tensor to fine polygon indicating text bounding boxes

        Parameters
        ----------
        hmap : torch.tensor
            Probability heatmap tensor.
        segmentation : torch.tensor
            Segmentataion tensor.
        dest_width : TYPE
            target width of the output.
        dest_height : TYPE
            target width of the output.
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
        
        Returns
        -------
        boxes_batch : list of lists
            Polygon bounding boxes of each text box.
        scores_batch : list of floats
            Confidence scores of each text box.
        '''        
        assert segmentation.size(0) == 1
        bitmap = segmentation.cpu().numpy()[0]  # The first channel
        hmap = hmap.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
                            (bitmap*255).astype(np.uint8),
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if max_candidates > 0:
            num_contours = min(len(contours), max_candidates)
        else:
            num_contours = len(contours)

        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
    
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < bbox_min_size:
                continue

            points = np.array(points)
            score = self.box_score_fast(hmap, points.reshape(-1, 2))
            if score < bbox_min_score:
                continue
        
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < bbox_min_size + 2:
                continue

            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score

        return boxes.tolist(), scores
    
    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))

        return expanded
    
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
    
        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]

        return box, min(bounding_box[1])
    
    def box_score_fast(self, hmap, box_):
        '''
        Calculate total score of each bounding box

        Parameters
        ----------
        hmap : torch.tensor
            Probability heatmap tensor.
        box_ : list
            Rectanguar bounding box.

        Returns
        -------
        float
            Confidence score.
        '''
        h, w = hmap.shape[:2]
        box = box_.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
    
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(hmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
    
    def image2hmap(self, image_tensor):
        '''
        Run the model to obtain a heatmap tensor from a image tensor. The heatmap
        tensor indicates the probability of each pixel being a part of text area.

        Parameters
        ----------
        image_tensor : torch.tensor
            Image tensor.

        Returns
        -------
        torch.tensor
            Probability heatmap tensor.
        '''
        return self.model.forward(image_tensor, training=False)
        
    def inference(self, 
                  image,
                  text_threshold = 0.2, 
                  bbox_min_score = 0.2, 
                  bbox_min_size = 3, 
                  max_candidates = 0, 
                  detection_size = None,
                  as_polygon = False,
                  return_scores = False):
        '''
        Wrapper to run the model on an input image to get text bounding boxes.

        Parameters
        ----------
        image : path-to-file, PIL.Image, or np.ndarray
            Image to load or convert.
        text_threshold : float, optional
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
        detection_size : int, optional
            Target detection size. Please see docstring under method resize_image()
            for explanation. The default is None.
        as_polygon : boolean, optional
            If true, return the bounding boxes as find polygons, otherwise, return
            as rectagular. The default is False.
        return_scores : boolean, optional
            If true, return confidence score along with the text bounding boxes.
            The default is False.

        Returns
        -------
        list of lists
            Text bounding boxes. If return_scores is set to true, another list
            of lists will also be returned.

        '''
        if not isinstance(image, list):
            image = [image]

        image_tensor, original_shapes = self.load_images(image, detection_size = detection_size)
        with torch.no_grad():
            hmap = self.image2hmap(image_tensor)
            batch_boxes, batch_scores = self.hmap2bbox(image_tensor, 
                                                       original_shapes,
                                                       hmap, 
                                                       text_threshold = text_threshold, 
                                                       bbox_min_score = bbox_min_score, 
                                                       bbox_min_size = bbox_min_size, 
                                                       max_candidates = max_candidates, 
                                                       as_polygon=as_polygon) 
        
        if return_scores:
            return batch_boxes, batch_scores
        else:
            return batch_boxes
    
