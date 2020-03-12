from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np



img = Image.open('./download.jpeg')
plt.imshow(img)
plt.show()

# Transformations
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

# Forward Pass
out = fcn(inp)['out']
print(out.shape)

# Checking max index
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(om.shape)
# Picture of dog has class pixel position
print(np.unique(om))


def decode_segmap(image, nc=21):

    label_colours = np.array([(0,0,0), (128,0,0), (0,128,0),
                              (128,128,0), (0,0,128), (128,0,128),
                              (0,128,128), (128,128,128), (64,0,0),
                              (192,0,0), (64,128,0), (192,128,0), (64,0,128),
                              (192,0,128), (64,128,128), (192,128,128), (0,64,0),
                              (128,64,0), (0,192,0), (128,192,0), (0,64,128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colours[l, 0]
        g[idx] = label_colours[l, 1]
        b[idx] = label_colours[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path):
    img = Image.open(path)
    plt.imshow(img); plt.axis('off'); plt.show()

    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb); plt.axis('off'); plt.show()


segment(fcn, './cat.jpeg')


