import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm


def find_best_sliding_window_match(image, kernel, with_padding=false):
    """
    Calculate best position on image given pattern.
    :param image: RGB image
    :param kernel: RGB image
    :param mode: same / valid whether to compute similarity on borders
    :return: best position, best_score
    """
    image_H, image_W, image_channels = image.shape
    kernel_H, kernel_W, kernel_channels = kernel.shape
    assert image_channels == kernel_channels, "Image number of channels should be equal to kernel ones"

    current_best_score = float("inf")
    current_best_position = []
    height_limit = image_H
    width_limit = image_W

    if not with_padding:
        height_limit -= kernel_H - 1
        width_limit -= kernel_W - 1
        search_image = image
    else:
        search_image = np.pad(image, ((kernel_H // 2, kernel_H // 2), (kernel_W // 2, kernel_W // 2), (0, 0)))
        height_limit += kernel_H // 2
        width_limit += kernel_W // 2

    for ii in tqdm(range(height_limit)):
        for jj in range(width_limit):
            this_score = np.sum(np.abs(search_image[ii: ii + kernel_H, jj: jj + kernel_W, :] - kernel))
            if this_score < current_best_score:
                current_best_position = [ii, jj]
                current_best_score = this_score

    if with_padding:
        current_best_position[0] -= kernel.shape[0] - 1
        current_best_position[1] -= kernel.shape[1] - 1
    return current_best_position, current_best_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path in input image for pattern search", default="../images/target.jpeg")
    parser.add_argument("-q", "--query", help="Pattern to search for", default="../images/query0.jpg")
    parser.add_argument("-v", "--pad", help="Use same padding",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    image = np.array(Image.open(arguments.image))
    kernel = np.array(Image.open(arguments.query))

    best_position, best_score = find_best_sliding_window_match(image, kernel, arguments.pad)
    print(best_position, best_score)
