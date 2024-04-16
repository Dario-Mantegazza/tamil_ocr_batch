import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# import craft_text_detector.craft_utils as craft_utils
# import craft_text_detector.image_utils as image_utils
# import craft_text_detector.torch_utils as torch_utils

from . import craft_utils as craft_utils
from . import image_utils as image_utils
from . import torch_utils as torch_utils

class CRAFT_Dataset(Dataset):
    def __init__(self, imgs, canvas_size, use_cuda, half):
        self.imgs = imgs
        self.canvas_size = canvas_size
        self.use_cuda= use_cuda
        self.half= half

    def transform(self, img):
        # resize
        img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
            img, self.canvas_size, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = image_utils.normalizeMeanVariance(img_resized)
        x = torch_utils.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        # x = torch_utils.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.use_cuda:
            if self.half:
                x = x.cuda().half()
            else:
                x = x.cuda()
        return x, ratio_h, ratio_w
    


    def __getitem__(self, index):
        img = self.imgs[index]
        img = np.array(img)
        img, ratio_h, ratio_w = self.transform(img)
        return img, ratio_h, ratio_w

    def __len__(self):
        return len(self.imgs)

def get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    long_size: int = 1280,
    poly: bool = True,
    half: bool = False,
    craft_batch_size=1,
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """

    data=CRAFT_Dataset(image, long_size, cuda, half=half)
    dataloader=DataLoader(data, batch_size=craft_batch_size)
    # forward pass
    y=None
    with torch.inference_mode():
        for batch in dataloader:
            x, ratio_h, ratio_w = batch
            if y is None:
                y, _ = craft_net(x)
            else:
                y = torch.cat((y, craft_net(x)[0]), 0)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = y[0, :, :, 1].cpu().data.numpy().astype(np.float32)


    # Post-processing
    boxes = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h) # probabilemente non va causa batch di ratios

    return boxes
