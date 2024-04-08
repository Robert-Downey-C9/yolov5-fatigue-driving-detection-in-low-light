import numpy as np
from PIL import Image


'''
    数据扩增7种模式
    0-原图
    1-上下翻转
    2-逆时针旋转90
    3-逆时针旋转90，再上下翻转
    4-逆时针旋转180
    5-逆时针旋转270
    6-逆时针旋转270，再上下翻转
'''

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

'''
    加载图像，得到像素值归一化后的图像数据
'''

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

'''
    保存结果图像（可保存两个图像拼接后的图像，便于视觉效果上进行对比）。
'''

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
