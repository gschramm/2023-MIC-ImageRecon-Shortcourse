import torch
import collections
from array_api_compat import to_device
from utils import distributed_subset_order


def sequential_conv_model(device,
                          kernel_size=(3, 3, 3),
                          num_layers=6,
                          num_features=10,
                          batch_norm: bool = False,
                          dtype=torch.float32) -> torch.nn.Sequential:
    """simple sequential model consisting of 3D conv layers and PReLUs

    Parameters
    ----------
    device : optional
        by default torch.device("cuda:0")
    kernel_size : tuple, optional
        kernel size of conv layers, by default (3, 3, 1)
    num_layers : int, optional
        number of conv layers, by default 6
    num_features : int, optional
        number of features, by default 10
    batch_norm : bool, optional
        use batch norm, by default False
    dtype : optional
        data type for conv layers, by default torch.float32

    Returns
    -------
    Sequential model
    """

    conv_net = collections.OrderedDict()

    conv_net['conv_1'] = torch.nn.Conv3d(1,
                                         num_features,
                                         kernel_size,
                                         padding='same',
                                         device=device,
                                         dtype=dtype)

    if batch_norm:
        conv_net['batch_norm_1'] = torch.nn.BatchNorm3d(num_features,
                                                        device=device)

    conv_net['prelu_1'] = torch.nn.PReLU(device=device)

    for i in range(num_layers - 2):
        conv_net[f'conv_{i+2}'] = torch.nn.Conv3d(num_features,
                                                  num_features,
                                                  kernel_size,
                                                  padding='same',
                                                  device=device,
                                                  dtype=dtype)

        if batch_norm:
            conv_net[f'batch_norm_{i+2}'] = torch.nn.BatchNorm3d(num_features,
                                                                 device=device)

        conv_net[f'prelu_{i+2}'] = torch.nn.PReLU(device=device)

    conv_net[f'conv_{num_layers}'] = torch.nn.Conv3d(num_features,
                                                     1,
                                                     kernel_size,
                                                     padding='same',
                                                     device=device,
                                                     dtype=dtype)
    conv_net[f'prelu_{num_layers}'] = torch.nn.PReLU(device=device)

    conv_net = torch.nn.Sequential(conv_net)

    return conv_net


class DoubleConv3DBlock(torch.nn.Module):
    """convolution, batch norm, relu, convolution, batch norm, relu"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._double_conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=3,
                            padding='same'),
            torch.nn.BatchNorm3d(out_channels), torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_channels,
                            out_channels,
                            kernel_size=3,
                            padding='same',
                            bias=False), torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        return self._double_conv(x)


class Unet3DDownBlock(torch.nn.Module):
    """maxpool downsampling followed by double conv block"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool3d(2), DoubleConv3DBlock(in_channels,
                                                     out_channels))

    def forward(self, x):
        return self._maxpool_conv(x)


class Unet3DUpBlock(torch.nn.Module):
    """bilinear upsampling, concatenation, double conv block"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self._up = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        self._conv = DoubleConv3DBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, self._up(x1)], dim=1)
        return self._conv(x)


class Unet3dFinalConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self._conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self._conv(x)


class Unet3D(torch.nn.Module):
    """3D Unet with 3D downsampling and upsampling blocks"""

    def __init__(self, num_features: int = 8, num_input_channels: int = 1):
        super().__init__()
        self._num_features = num_features
        self._num_input_channels = num_input_channels

        self.first_double_conv = (DoubleConv3DBlock(self._num_input_channels,
                                                    self._num_features))
        self.down1 = (Unet3DDownBlock(self._num_features,
                                      2 * self._num_features))
        self.down2 = (Unet3DDownBlock(2 * self._num_features,
                                      4 * self._num_features))
        self.down3 = (Unet3DDownBlock(4 * self._num_features,
                                      4 * self._num_features))
        self.up1 = (Unet3DUpBlock(8 * self._num_features,
                                  2 * self._num_features))
        self.up2 = (Unet3DUpBlock(4 * self._num_features,
                                  1 * self._num_features))
        self.up3 = (Unet3DUpBlock(2 * self._num_features, self._num_features))
        self.final_conv = Unet3dFinalConv(self._num_features, 1)

    def forward(self, x):
        x1 = self.first_double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        return self.final_conv(y)


if __name__ == '__main__':
    import tempfile
    from torch.utils.tensorboard import SummaryWriter
    dev = "cpu"
    dtype = torch.float32

    x = torch.rand(4, 1, 128, 128, 16, dtype=dtype).to(dev)

    model = Unet3D(num_features=8)
    y = model(x)
    print('number of trainable parameters:',
          sum(p.numel() for p in model.parameters()))

    tmp_run_dir = tempfile.TemporaryDirectory()
    writer = SummaryWriter(tmp_run_dir.name)
    writer.add_graph(model, x)
    writer.close()

    print(
        f'run "tensorboard --logdir {tmp_run_dir.name}" to view model in tensorboard'
    )

class SimpleOSEMVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 osem_update_modules: torch.nn.Module,
                 neural_net: torch.nn.Module,
                 depth: int,
                 device) -> None:

        super().__init__()

        self._osem_update_modules = osem_update_modules

        self._num_subsets = len(osem_update_modules)
        self._subset_order = distributed_subset_order(self._num_subsets)

        self._neural_net = neural_net
        self._depth = depth

        self._neural_net_weight = torch.nn.ParameterList(
            [torch.ones(1, device=device) for _ in range(self._depth)])

    @property
    def neural_net_weight(self) -> torch.Tensor:
        return self._neural_net_weight

    @property
    def neural_net(self) -> torch.nn.Module:
        return self._neural_net

    def forward(self, x: torch.Tensor, emission_data_batch: torch.Tensor,
                correction_batch: torch.Tensor,
                contamination_batch: torch.Tensor,
                adjoint_ones_batch: torch.Tensor) -> torch.Tensor:

        for j in range(self._depth):
            subset = self._subset_order[j % self._num_subsets]
            x_em = self._osem_update_modules[subset](
                x, emission_data_batch[subset, ...], correction_batch[subset,
                                                                      ...],
                contamination_batch[subset, ...], adjoint_ones_batch[subset,
                                                                     ...])

            x_nn = self._neural_net(x)

            # fusion of EM update and neural net update with trainable weight
            # we use an ReLU activation to ensure that the output of each block is non-negative
            x = torch.nn.ReLU()(x_em + self._neural_net_weight[j] * x_nn)

        return x

class PostReconNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 neural_net: torch.nn.Module) -> None:
        super().__init__()
        self._neural_net = neural_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fusion of EM update and neural net update with trainable weight
        # we use an ReLU activation to ensure that the output of each block is non-negative
        return torch.nn.ReLU()(x + self._neural_net(x))

