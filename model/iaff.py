from torch import ones_like, Tensor
from torch.functional import F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Module, ReLU, Sequential
from torch.nn.init import constant_, kaiming_normal_


class IAFF(Module):
    def __init__(self, num_in_channels: int, gamma: int) -> None:
        super(IAFF, self).__init__()
        num_out_channels = num_in_channels // gamma
        self._local_attention_1 = Sequential(
            Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=num_out_channels, out_channels=num_in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_in_channels)
        )
        self._global_attention_1 = Sequential(
            AdaptiveAvgPool2d(output_size=1),
            Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=num_out_channels, out_channels=num_in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_in_channels)
        )
        self._local_attention_2 = Sequential(
            Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=num_out_channels, out_channels=num_in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_in_channels)
        )
        self._global_attention_2 = Sequential(
            AdaptiveAvgPool2d(output_size=1),
            Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=num_out_channels, out_channels=num_in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=num_in_channels)
        )
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(tensor=module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    constant_(tensor=module.bias, val=0)
            elif isinstance(module, BatchNorm2d):
                constant_(tensor=module.weight, val=1)
                constant_(tensor=module.bias, val=0)

    def forward(self, shallow_feature: Tensor, deep_feature: Tensor) -> Tensor:
        output = shallow_feature + deep_feature
        output = F.sigmoid(input=self._local_attention_1(input=output) + self._global_attention_1(input=output))
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        output = F.sigmoid(input=self._local_attention_2(input=output) + self._global_attention_2(input=output))
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        return output
