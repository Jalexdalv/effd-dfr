from torch import cat, Tensor
from torch.nn import BatchNorm2d, Conv2d, Identity, Module, ModuleList, ReLU, Sequential
from torch.nn.init import constant_, kaiming_normal_


class CAE(Module):
    def __init__(self, num_in_channels: int, alpha: int, betas: tuple) -> None:
        super(CAE, self).__init__()
        channels = []
        self._encoder = ModuleList()
        for index in range(alpha):
            num_out_channels = num_in_channels // betas[index]
            self._encoder.append(Sequential(
                Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, bias=False),
                BatchNorm2d(num_features=num_out_channels),
                ReLU(inplace=True)
            ))
            channels.append(num_out_channels)
            self._encoder.append(Sequential(
                Conv2d(in_channels=num_out_channels, out_channels=num_out_channels, kernel_size=1, bias=index == alpha - 1),
                BatchNorm2d(num_features=num_out_channels) if index > alpha - 1 else Identity(),
                ReLU(inplace=True) if index > alpha - 1 else Identity()
            ))
            channels.append(num_out_channels)
            num_in_channels = num_out_channels
        self._decoder = ModuleList()
        for index in reversed(range(alpha)):
            num_out_channels = num_in_channels * betas[index]
            self._decoder.append(Sequential(
                Conv2d(in_channels=num_in_channels + channels.pop(), out_channels=num_in_channels, kernel_size=1, bias=False),
                BatchNorm2d(num_features=num_in_channels),
                ReLU(inplace=True)
            ))
            self._decoder.append(Sequential(
                Conv2d(in_channels=num_in_channels + channels.pop(), out_channels=num_out_channels, kernel_size=1, bias=index == 0),
                BatchNorm2d(num_features=num_out_channels) if index > 0 else Identity(),
                ReLU(inplace=True) if index > 0 else Identity()
            ))
            num_in_channels = num_out_channels
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(tensor=module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    constant_(tensor=module.bias, val=0)
            elif isinstance(module, BatchNorm2d):
                constant_(tensor=module.weight, val=1)
                constant_(tensor=module.bias, val=0)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        features = []
        for layer in self._encoder:
            output = layer(input=output)
            features.append(output)
        for layer in self._decoder:
            output = layer(input=cat(tensors=[output, features.pop()], dim=1))
        return output
