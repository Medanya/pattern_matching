import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pattern_search import calculate_pattern_heatmap, parse_arguments, get_argmin_position


def visualize_pattern_search(image, query, with_padding):
    scores = calculate_pattern_heatmap(image, query, with_padding)
    answer = get_argmin_position(scores)
    if with_padding:
        answer[0] -= kernel.shape[0] - 1
        answer[1] -= kernel.shape[1] - 1


    if not with_padding:
        scores = np.pad(scores, [(kernel.shape[0] // 2, 0), (kernel.shape[1] // 2, 0)], constant_values=scores.max())
        rect = patches.Rectangle((answer[1], answer[0]), query.shape[1], query.shape[0], linewidth=1, edgecolor='r',
                                 facecolor='none')
    else:
        image = np.pad(image, [(kernel.shape[0] // 2, kernel.shape[0] // 2), (kernel.shape[1] // 2, kernel.shape[0] // 2), (0, 0)], constant_values=255)
        rect = patches.Rectangle((answer[1] + kernel.shape[1] // 2, answer[0] + kernel.shape[0] // 2), query.shape[1], query.shape[0], linewidth=1, edgecolor='r',
                                 facecolor='none')

    fig, axs = plt.subplots(1, 1)
    axs.set_title("heatmap scores")
    axs.imshow(image)
    cs = axs.contour(scores, np.percentile(scores.ravel(), [5, 10, 20, 30, 40, 50]), linewidths=2, cmap=plt.cm.Greens, alpha=0.75)
    plt.colorbar(cs, extend='both')
    axs.add_patch(rect)
    plt.axis("off")
    plt.figure(2)
    plt.title("query")
    plt.imshow(query)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    arguments = parse_arguments()
    image = np.array(Image.open(arguments.image))
    kernel = np.array(Image.open(arguments.query))
    visualize_pattern_search(image, kernel, arguments.pad)