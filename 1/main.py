import numpy as np
from PIL import Image
from tqdm import tqdm


def find_best_sliding_window_match(image, kernel):
    """
    Calculate best position on image given pattern.
    :param image: RGB image
    :param kernel: RGB image to search for
    :return: best match position, best match score
    """
    image_H, image_W, image_channels = image.shape
    kernel_H, kernel_W, kernel_channels = kernel.shape
    assert image_channels == kernel_channels, "Image number of channels should be equal to kernel ones"
    current_best_score = float("inf")
    current_best_position = []

    for ii in tqdm(range(image_H - kernel_H + 1)):
        for jj in range(image_W - kernel_W + 1):
            this_score = np.sum(np.abs(image[ii: ii + kernel_H, jj: jj + kernel_W, :] - kernel))
            if this_score < current_best_score:
                current_best_position = [ii, jj]
                current_best_score = this_score
    return current_best_position, current_best_score


if __name__ == "__main__":
    image = np.array(Image.open("../images/target.jpeg"))
    kernel = np.array(Image.open("../images/query0.jpg"))

    best_position, best_score = find_best_sliding_window_match(image, kernel)
    print("Best pattern position: {}, it's score: {}".format(best_position, best_score))
