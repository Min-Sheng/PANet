"""
Customized data transforms
"""
import random
import math
import numbers
from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        h, w = img.size[::-1]
        new_h, new_w = self.size

        if new_h < h and  new_w < w:
            img = tr_F.resize(img, self.size[::-1], interpolation=Image.ANTIALIAS)
        else:
            img = tr_F.resize(img, self.size[::-1], interpolation=Image.BILINEAR)

        label = tr_F.resize(label, self.size[::-1], interpolation=Image.NEAREST)
        inst = tr_F.resize(inst, self.size[::-1], interpolation=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample

class RandomResize(object):
    """Resize a tuple of images to a random size and aspect ratio.
    Args:
        scale (list, optional): The range of size of the origin size (default: 0.75 to 1.33).
        ratio (list, optional): The range of aspect ratio of the origin aspect ratio before the sqrt operation(default: 2/3, 3/2)
        prob  (float, optional): The probability of applying the resize (default: 0.5).
    """
    def __init__(self, scale=[0.75, 1.33], ratio=[2./3., 3./2.], prob=0.5):
        if len(scale) != 2:
            raise ValueError("Scale must be a sequence of len 2.")
        if len(ratio) != 2:
            raise ValueError("ratio must be a sequence of len 2.")
        self.scale = scale
        self.ratio = ratio
        self.prob = max(0, min(prob, 1))

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        h, w = img.size[::-1]
        area = h * w
        random_scale = random.uniform(*self.scale)
        target_area = random_scale * area

        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        random_ratio = math.exp(random.uniform(*log_ratio))
        new_h = int(round(math.sqrt(target_area * random_ratio)))
        new_w = int(round(math.sqrt(target_area * random_ratio)))
        new_size = (new_h,new_w)

        if random.random() < self.prob:
            if new_h < h and  new_w < w:
                img = tr_F.resize(img, new_size[::-1], interpolation=Image.ANTIALIAS)
            else:
                img = tr_F.resize(img, new_size[::-1], interpolation=Image.BILINEAR)
            label = tr_F.resize(label, new_size[::-1], interpolation=Image.NEAREST)
            inst = tr_F.resize(inst, new_size[::-1], interpolation=Image.NEAREST)
        
        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample

class RandomRotation(object):
    """
    Rotate the images/masks to a random degree
    Args:
        degrees (list, optional): The range of degrees to select from (default: -30 to 30).
            If degrees is a nmber instead of a list like [min, max], the range of degrees will be [-degree, degree]
        prob  (float, optional): The probability of applying the resize (default: 0.5).
    """
    def __init__(self, degrees=[-30, 30], prob=0.5):
        
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = [-degrees, degrees]
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.prob = prob

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        if random.random() < self.prob:
            angle = random.uniform(*self.degrees)
            img = img.rotate(angle, Image.BILINEAR)
            label = label.rotate(angle, Image.NEAREST)
            inst = inst.rotate(angle, Image.NEAREST)
        
        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample

class RandomHorizontalFlip(object):
    """
    Randomly filp the images/masks horizontally
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
        
        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample

class RandomVerticalFlip(object):
    """
    Randomly filp the images/masks vertically
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
            inst = inst.transpose(Image.FLIP_TOP_BOTTOM)
        
        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample

class RandomCrop(object):
    """Crop the images/masks at the same random location.
    Args:
        size (list): The desired output size of the cropped images/masks.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        
        img, label = sample['image'], sample['label']
        inst = sample['inst']

        new_img_h0, new_img_hn, new_img_w0, new_img_wn, img_h0, img_hn, img_w0, img_wn = self._get_coordinates(img, self.size)
        
        cropped_img = img.crop((img_w0, img_h0, img_wn, img_hn))
        cropped_label = label.crop((img_w0, img_h0, img_wn, img_hn))
        cropped_inst = inst.crop((img_w0, img_h0, img_wn, img_hn))
        
        new_img = Image.new(mode = "RGB", size = self.size[::-1])
        new_label = Image.new(mode = "L", size = self.size[::-1])
        new_inst = Image.new(mode = "I", size = self.size[::-1])
        new_img.paste(cropped_img, (new_img_w0, new_img_h0, new_img_wn, new_img_hn))
        new_label.paste(cropped_label, (new_img_w0, new_img_h0, new_img_wn, new_img_hn))
        new_inst.paste(cropped_inst,(new_img_w0, new_img_h0, new_img_wn, new_img_hn))

        sample['image'] = new_img
        sample['label'] = new_label
        sample['inst'] = new_inst

        return sample

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (PIL.Image): The image to be cropped.
        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
       
        h, w = img.size[::-1]
        ht, wt = min(size[0], h), min(size[1], w)
        h_space, w_space = h - size[0], w -size[1]
        
        if h_space > 0:
            new_img_h0 = 0
            img_h0 = random.randrange(h_space + 1)
        else:
            new_img_h0 = random.randrange(-h_space + 1)
            img_h0 = 0
        if w_space > 0:
            new_img_w0 = 0
            img_w0 = random.randrange(w_space + 1)
        else:
            new_img_w0 = random.randrange(-w_space + 1)
            img_w0 = 0

        return new_img_h0, new_img_h0 + ht, new_img_w0, new_img_w0 + wt, img_h0, img_h0 + ht, img_w0, img_w0 + wt

    
class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        if means is None and stds is None:
            pass
        elif means is not None and stds is not None:
            if len(means) != len(stds):
                raise ValueError('The number of the means should be the same as the standard deviations.')
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

        self.means = means
        self.stds = stds
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst = sample['inst']
        img = tr_F.to_tensor(img) # rescale to 0~1
        if self.means is None and self.stds is None: # Apply image-level normalization.
            self.means = [img.mean().item()] * img.size(0)
            self.stds = [img.std().item()] * img.size(0)
        img = tr_F.normalize(img, mean=self.means, std=self.stds)
        label = torch.Tensor(np.array(label)).long()
        inst = torch.Tensor(np.array(inst)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        return sample