import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    confmat: np.ndarray,
    class_names,
    save_path=None,
    normalize=False,
    cmap="Blues",
    bg_color="white",
):
    """
    Plot and save confusion matrix as image.
    Args:
        confmat (tensor): Confusion matrix tensor.
        class_names (list of str): Class names.
        normalize (bool): If True, normalize the confusion matrix.
        cmap (str): Colormap for the plot.
        bg_color (str): Background color of the plot.

    """

    if normalize:
        confmat = confmat / confmat.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(
        figsize=(len(class_names), len(class_names)), facecolor=bg_color
    )
    ax.imshow(confmat, cmap=cmap)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = np.nan_to_num(confmat).max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if confmat[i, j] > threshold else "black"
            text = f"{confmat[i,j]:.2f}" if normalize else f"{confmat[i,j]}"
            ax.text(j, i, text, ha="center", va="center", color=color)

    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, facecolor=bg_color)  # save figure with epoch number
    plt.close()
