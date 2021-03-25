import torchvision.models as models
import cv2
import torch
import torchvision.transforms as t
import numpy as np
import torch.nn as nn
from PIL import Image


def preprocess_image(img, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        ims_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(img, (224, 224))
    im_as_arr = np.float32(img)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = torch.autograd.Variable(im_as_ten, requires_grad=True)
    return im_as_var

class Net(nn.Module):
    """
    init resnet18
    you can change the net here
    """
    def __init__(self):
        super(Net, self).__init__()
        self.Net = models.resnet50(pretrained=True)

    def forward(self, input):
        output = self.Net.conv1(input)
        output = self.Net.bn1(output)
        output = self.Net.relu(output)
        output = self.Net.maxpool(output)
        output = self.Net.layer1(output)
        output = self.Net.layer2(output)
        output = self.Net.layer3(output)
        output = self.Net.layer4(output)
        output = self.Net.avgpool(output)
        return output

def extractor(img_path, saved_path, net, use_gpu=True):
    transform = t.Compose([
        t.Resize(256),
        t.CenterCrop(224),
        t.ToTensor()
    ])

    img = cv2.imread(img_path)
    #img = Image.open(img_path)
    #img = transform(img)
    img = preprocess_image(img)
    print(img.shape)

    x = img
    #x = torch.autograd.Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    #print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    outputimg = net(x).cpu()
    print("outputimg.shape:")
    print(outputimg.shape)

    outputimg = outputimg[:, 0, :, :]
    print(outputimg.shape)

    outputimg = outputimg.view(outputimg.shape[1], outputimg.shape[2])
    print(outputimg.shape)

    outputimg = outputimg.data.numpy()
    # use sigmod to [0,1]
    output = 1.0 / (1 + np.exp(-1 * outputimg))

    # to [0,255]
    output = np.round(output * 255)
    print(output[0])

    cv2.imwrite(saved_path, output)

if __name__ == '__main__':
    model = Net()
    model = model.eval().cuda()

    img_path = './image/dog.jpg'
    saved_path = './image/dog_resnet.jpg'

    use_gpu = torch.cuda.is_available()

    extractor(img_path, saved_path, model, use_gpu)




