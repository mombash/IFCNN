import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme  # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        """
        Pads a list of tensors to the specified padding dimensions.

        Args:
            tensors (list of torch.Tensor): List of tensors to pad.
            padding (tuple): Padding dimensions (e.g., (left, right, top, bottom)).
            mode (str): Padding mode ('constant', 'reflect', 'replicate').
            value (float): Padding value (used only for 'constant' mode).

        Returns:
            list of torch.Tensor: List of padded tensors.
        """
        out_tensors = []
        for i, tensor in enumerate(tensors):
            print(f"Tensor {i} original shape: {tensor.shape}")  # Debug: Print the original shape

            # Handle 5D tensors (batch, channels, height, width, depth)
            if tensor.dim() == 5:
                print(f"Tensor {i} is 5D. Slicing along the depth axis...")
                slices = []
                for d in range(tensor.shape[-1]):  # Iterate over the depth dimension
                    slice_2d = tensor[..., d]  # Extract a 2D slice
                    slices.append(slice_2d)
                tensor = torch.cat(slices, dim=0)  # Combine slices into a batch of 2D images
                print(f"Tensor {i} reshaped to: {tensor.shape}")

            # Ensure the tensor is 4D (batch, channels, height, width)
            if tensor.dim() == 2:
                print(f"Tensor {i} is 2D. Reshaping to 4D...")
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif tensor.dim() == 3:
                print(f"Tensor {i} is 3D. Reshaping to 4D...")
                tensor = tensor.unsqueeze(0)  # Add batch dimension
            elif tensor.dim() != 4:
                print(f"Tensor {i} is not 4D. Current shape: {tensor.shape}")
                raise ValueError(f"Tensor {i} must be 4D for padding, but got shape {tensor.shape}")

            print(f"Tensor {i} reshaped to: {tensor.shape}")  # Debug: Print the reshaped tensor

            # Apply padding
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, t1, mra):
        # Ensure tensors are 4D or slice 5D tensors
        if t1.dim() == 5:
            print(f"T1 is 5D. Slicing along the depth axis...")
            t1_slices = [t1[..., d] for d in range(t1.shape[-1])]  # Slice along depth
            t1 = torch.cat(t1_slices, dim=0)  # Combine slices into a batch
            print(f"T1 reshaped to: {t1.shape}")
        if mra.dim() == 5:
            print(f"MRA is 5D. Slicing along the depth axis...")
            mra_slices = [mra[..., d] for d in range(mra.shape[-1])]  # Slice along depth
            mra = torch.cat(mra_slices, dim=0)  # Combine slices into a batch
            print(f"MRA reshaped to: {mra.shape}")

        # Replicate single-channel tensors to 3 channels
        if t1.shape[1] == 1:
            print("Replicating T1 to 3 channels...")
            t1 = t1.repeat(1, 3, 1, 1)
        if mra.shape[1] == 1:
            print("Replicating MRA to 3 channels...")
            mra = mra.repeat(1, 3, 1, 1)

        tensors = [t1, mra]

        # Feature extraction with padding
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)

        # Feature fusion
        if self.fuse_scheme == 0:  # MAX
            out = self.tensor_max(outs)
        elif self.fuse_scheme == 1:  # SUM
            out = self.tensor_sum(outs)
        elif self.fuse_scheme == 2:  # MEAN
            out = self.tensor_mean(outs)
        else:  # Default: MAX
            out = self.tensor_max(outs)

        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out


def myIFCNN(fuse_scheme=0):
    # pretrained resnet101
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    # our model
    model = IFCNN(resnet, fuse_scheme=fuse_scheme)
    return model