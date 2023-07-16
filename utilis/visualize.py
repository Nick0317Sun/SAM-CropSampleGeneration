import numpy as np
import matplotlib.pyplot as plt


def normalize(array):
    """The direct visualization effect of Satellite imagery is often not ideal, and 2% stretching is carried out"""
    array_min = np.percentile(array, 2)
    array_max = np.percentile(array, 98)

    # array_min = np.min(array)
    # array_max = np.max(array)

    array[np.where(array < array_min)] = array_min
    array[np.where(array > array_max)] = array_max

    result = (array - array_min) / (array_max - array_min)

    return result


def show_points(coords, labels, ax):
    # garlic = coords[np.where(labels == 2)]
    # wheat = coords[np.where(labels == 1)]
    # others = coords[np.where(labels == 0)]
    # ax.scatter(garlic[:, 1], garlic[:, 0], color='red')
    # ax.scatter(wheat[:, 1], wheat[:, 0], color='green')
    # ax.scatter(others[:, 1], others[:, 0], color='blue')

    ax.scatter(coords[:, 1], coords[:, 0], color='red', s=20)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
        # color = np.array([0 / 255, 191 / 255, 255 / 255, 0.6])
        # color = np.array([0 / 255, 255 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def vis_pointInMap(map, points, labels, save_path):
    """
    visualize point in the map
    :param map:  image [h, w, c]
    :param points:  point [point1, point2, ...]
    :param save_path:
    :return:
    """
    # plt.imshow(map)
    plt.imshow(normalize(map))
    show_points(points, labels, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, transparent=True)
    plt.close()


def vis_MaskInMap(map, points, labels, mask, mark, score, save_path):
    """
        visualize mask in the map
    """
    # plt.imshow(map)
    plt.imshow(normalize(map))
    show_mask(mask, plt.gca())
    show_points(points, labels, plt.gca())
    plt.title(f"Mask {mark + 1}, Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(save_path, transparent=True)
    plt.close()


def vis_SITS(SITSs, save_path, reference=None):
    """
    可视化时间序列
    :param SITS:
    :param types: 指明是原始样本还是SAM生成的样本
    :param save_path:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    for j in range(SITSs.shape[0]):  #
        ax.plot(SITSs[j, :], color='grey')
    if not reference is None:
        ax.plot(reference[0, :], color='red')
    plt.tight_layout()
    plt.savefig(save_path, transparent=True)
    plt.close()
