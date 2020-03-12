from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# GPU checker - will use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# helper function to load the image
def image_loader(image_name, imsize):

    # resizes the image to appropriate size and transforms the image into tensor
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# Shows the image
def imshow(tensor, title=None):
    # To reconvert the tensor back to an image at the end
    unloader = transforms.ToPILImage()

    # clones the tensor so it doesn't make changes on the original image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)


# Helper function to find the Style Loss
def gram_matrix(input):
    a, b, c, d = input.size()
#   a = batch size
#   b = number of feature maps
#   c, d = dimensions of the feature map N = c*d

    features = input.view(a * b, c * d)

#   finds the gram product
    G = torch.mm(features, features.t())
    # normalises the gram matrix by dividing by the number of elements in each feature map

    return G.div(a * b * c * d)


# Content Loss Class
class ContentLoss(nn.Module):
    def __init__(self, target):
        # detaches the target content to compute the gradient
        # this is a stated value
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    # finds the loss between the input and the target
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# Style Loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# Helper class for the vgg network
# Create a module to normalise input image, ready to be put in sequential module
class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        # view the mean and std to work directly with the tensor image shape
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalise img
        return (img - self.mean) / self.std



# Special values for the mean and standard deviation provided by the research paper
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# layers to compute style/content losses
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # list for iteratable access
    content_losses = []
    style_losses = []

    # creates a new sequential network for modules that are activated sequentially
    model = nn.Sequential(normalization)

    i = 0 # counter for the conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv {}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}'.format(i)

            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # adds content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # adds style loss
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trims the the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# Gradient Descent
def get_input_optimizer(input_img):
    # shows that the input needs a gradient
    optimizer = optim.LBFGS([input_img.requires_grad()])
    return optimizer


# size of image depending if GPU is available or not
imsize = 512 if torch.cuda.is_available() else 128

num_steps = 300
# loading the style and content images
style_img = image_loader()
content_img = image_loader()

# Sets the plotting to be interactive
plt.ion()

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

# importing the vgg 19 model and set it in evaluation mode
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# The starting image and output
input_img = content_img.clone()
# White noise instead uncomment to use
# input_img = torch.randn(content_img.data.size(), device=device)

# Shows the input image
plt.figure()
imshow(input_img, title='Input Image')


def run_style_transfer(model, style_losses, content_losses,
                       input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of the updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            # prints information each 50 epoch
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print("Style loss : {:4f} Content loss: {:4f}".format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


print('Building style transfer model...')
model, style_losses, content_losses = get_style_model_and_losses(cnn, cnn_normalization_mean,
                                                                 cnn_normalization_std, style_img, content_img)

output = run_style_transfer(model, style_losses, content_losses, input_img, num_steps=num_steps)

plt.figure()
imshow(output, title='Output image')

plt.ioff()
plt.show()










