import GPUtil
import cv2
import matplotlib.pyplot as plt
import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Using cuda: ", torch.cuda.get_device_name())

    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type(torch.FloatTensor)
        print("Using cpu")
    return device


def plot_image(image, mask=None, image_color=True, image_cmap=None, mask_color=True, mask_cmap=None):
    fig = plt.figure

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_cmap is not None:
        plt.imshow(image, interpolation="nearest", cmap=image_cmap)
    else:
        plt.imshow(image, interpolation="nearest")

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        if mask_color:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if mask_cmap is not None:
            plt.imshow(mask, alpha=0.5, interpolation="nearest", cmap=mask_cmap)
        else:
            plt.imshow(mask, alpha=0.5, interpolation="nearest")

    plt.show()


def free_gpu_cache(tensors, print_out=False):
    gpu = GPUtil.getGPUs()[0]

    if print_out:
        print("\n", "=" * 100, "\nBefore Clearing")
        GPUtil.showUtilization()

    for tensor in tensors:
        del tensor

    torch.cuda.empty_cache()

    if print_out:
        print("\nAfter Clearing")
        GPUtil.showUtilization()
