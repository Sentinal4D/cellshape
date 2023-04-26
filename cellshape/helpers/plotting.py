import io
import PIL
import torchvision
import matplotlib.pyplot as plt


def plot_to_image(fig):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after
    this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    # image_bites = fig.to_image(format="png")
    # print(image_bites)
    fig.savefig(buf, format="jpeg")
    buf.seek(0)
    # Turn plot into tensor to save on tensorboard
    # buf.seek(0)
    # plot_buf = buf
    image_tensor = PIL.Image.open((buf))
    image = torchvision.transforms.ToTensor()(image_tensor)
    return image


def plot_point_cloud(points):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="o", s=10)
    return fig
