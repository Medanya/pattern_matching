import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


def calculate_pattern_heatmap(image, kernel, with_padding=False):
    """
    Calculates per pattern distances for every position on image.
    :param image: RGB image
    :param kernel: RGB image
    :param mode: same / valid whether to compute similarity on borders
    :return: heatmap of scores
    """
    image_H, image_W, image_channels = image.shape
    kernel_H, kernel_W, kernel_channels = kernel.shape
    assert image_channels == kernel_channels, "Image number of channels should be equal to kernel ones"

    height_limit = image_H
    width_limit = image_W

    if not with_padding:
        height_limit -= kernel_H - 1
        width_limit -= kernel_W - 1
        search_image = image
    else:
        search_image = np.pad(image, ((kernel_H - 1, kernel_H - 1), (kernel_W - 1, kernel_W - 1), (0, 0)))
        height_limit += kernel_H - 1
        width_limit += kernel_W - 1

    scores = np.zeros((height_limit, width_limit))
    #def compute_positions(position):
    #    ii, jj = position
    #    scores[ii, jj] = np.sum(np.abs(search_image[ii: ii + kernel_H, jj: jj + kernel_W, :] - kernel))

    for ii in tqdm(range(height_limit)):
        for jj in range(width_limit):
            scores[ii, jj] = np.sum(np.abs(search_image[ii: ii + kernel_H, jj: jj + kernel_W, :] - kernel))
    return scores


def get_argmin_position(scores):
    return list(np.unravel_index(np.argmin(scores, axis=None), scores.shape))


def find_best_sliding_window_match(image, kernel, with_padding=False):
    """
    Calculate best position on image given pattern.
    :param image: RGB image
    :param kernel: RGB image
    :param mode: same / valid whether to compute similarity on borders
    :return: best position, best_score
    """
    scores = calculate_pattern_heatmap(image, kernel, with_padding)
    answer = get_argmin_position(scores)
    if with_padding:
        answer[0] -= kernel.shape[0] - 1
        answer[1] -= kernel.shape[1] - 1
    return answer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path in input image for pattern search", default="../images/target.jpeg")
    parser.add_argument("-q", "--query", help="Pattern to search for", default="../images/query0.jpg")
    parser.add_argument("-p", "--pad", help="Use same padding",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    image = np.array(Image.open(arguments.image))
    kernel = np.array(Image.open(arguments.query))

    best_position, best_score = find_best_sliding_window_match(image, kernel, arguments.pad)
    print(best_position, best_score)
