import PIL
from PIL import Image
PIL.PILLOW_VERSION = PIL.__version__

from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from imageio import imread, imsave
import numpy as np


# Creates the colour maps for each important classes such as Dog, Cat, Person
def decode_segmap(image, nc=21):

    # Class colours for the layers in this case they will be all black for binarization purposes
    label_colours = np.array([(255,255,255), (0,0,0), (0,0,0),
                              (0,0,0), (0,0,0), (0,0,0),
                              (0,0,0), (0,0,0), (0,0,0),
                              (0,0,0), (0,0,0), (0,0,0), (0,0,0),
                              (0,0,0), (0,0,0), (0,0,0), (0,0,0),
                              (0,0,0), (0,0,0), (0,0,0), (0,0,0)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # Assigns the matrix of the image with the label colours
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colours[l, 0]
        g[idx] = label_colours[l, 1]
        b[idx] = label_colours[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path, height, width):
    img = Image.open(path)
    #plt.imshow(img); plt.axis('off'); plt.show()

    # Resizes the image to the product's width and height and turns it into a tensor
    trf = T.Compose([T.Resize((height, width)),
                     T.ToTensor(),
                     T.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    #plt.imshow(rgb); plt.axis('off'); plt.show()
    imsave('Masks/auto.jpg', rgb)


#segment(fcn, './jayson.jpg')


