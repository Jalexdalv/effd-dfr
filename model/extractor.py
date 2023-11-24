from backbone.vgg import Vgg19
from torch import load, Tensor
from torch.nn import Module


class Extractor(Module):
    def __init__(self, backbone: Module, path: str) -> None:
        super(Extractor, self).__init__()
        self._backbone = backbone
        self._backbone.load_state_dict(state_dict=load(f=path))
        for param in self._backbone.parameters():
            param.requires_grad = False
        self.channels = []


class Vgg19Extractor(Extractor):
    def __init__(self, pool: str, padding_mode: str, use_relu: bool, layers: tuple) -> None:
        super(Vgg19Extractor, self).__init__(backbone=Vgg19(pool=pool, padding_mode=padding_mode), path='../dfr/pretrain/vgg19-dcbb9e9d.pth')
        layer_map = {'layer_1_1': 64, 'layer_1_2': 64,
                     'layer_2_1': 128, 'layer_2_2': 128,
                     'layer_3_1': 256, 'layer_3_2': 256, 'layer_3_3': 256, 'layer_3_4': 256,
                     'layer_4_1': 512, 'layer_4_2': 512, 'layer_4_3': 512, 'layer_4_4': 512,
                     'layer_5_1': 512, 'layer_5_2': 512, 'layer_5_3': 512, 'layer_5_4': 512}
        self._layers_1, self._layers_2, self._layers_3, self._layers_4, self._layers_5 = [], [], [], [], []
        self._channels_1, self._channels_2, self._channels_3, self._channels_4, self._channels_5 = [], [], [], [], []
        for layer in layers:
            if str.startswith(layer, 'layer_1'):
                self._layers_1.append(layer)
                self._channels_1.append(layer_map[layer])
            elif str.startswith(layer, 'layer_2'):
                self._layers_2.append(layer)
                self._channels_2.append(layer_map[layer])
            elif str.startswith(layer, 'layer_3'):
                self._layers_3.append(layer)
                self._channels_3.append(layer_map[layer])
            elif str.startswith(layer, 'layer_4'):
                self._layers_4.append(layer)
                self._channels_4.append(layer_map[layer])
            elif str.startswith(layer, 'layer_5'):
                self._layers_5.append(layer)
                self._channels_5.append(layer_map[layer])
        if self._channels_1:
            self.channels.append(self._channels_1)
        if self._channels_2:
            self.channels.append(self._channels_2)
        if self._channels_3:
            self.channels.append(self._channels_3)
        if self._channels_4:
            self.channels.append(self._channels_4)
        if self._channels_5:
            self.channels.append(self._channels_5)
        self._use_relu = use_relu
        self._features = self._backbone.features

    def forward(self, input: Tensor) -> list:
        conv_1_1 = self._features[0](input=input)
        relu_1_1 = self._features[1](input=conv_1_1)
        conv_1_2 = self._features[2](input=relu_1_1)
        relu_1_2 = self._features[3](input=conv_1_2)
        pool_1 = self._features[4](input=relu_1_2)
        conv_2_1 = self._features[5](input=pool_1)
        relu_2_1 = self._features[6](input=conv_2_1)
        conv_2_2 = self._features[7](input=relu_2_1)
        relu_2_2 = self._features[8](input=conv_2_2)
        pool_2 = self._features[9](input=relu_2_2)
        conv_3_1 = self._features[10](input=pool_2)
        relu_3_1 = self._features[11](input=conv_3_1)
        conv_3_2 = self._features[12](input=relu_3_1)
        relu_3_2 = self._features[13](input=conv_3_2)
        conv_3_3 = self._features[14](input=relu_3_2)
        relu_3_3 = self._features[15](input=conv_3_3)
        conv_3_4 = self._features[16](input=relu_3_3)
        relu_3_4 = self._features[17](input=conv_3_4)
        pool_3 = self._features[18](input=relu_3_4)
        conv_4_1 = self._features[19](input=pool_3)
        relu_4_1 = self._features[20](input=conv_4_1)
        conv_4_2 = self._features[21](input=relu_4_1)
        relu_4_2 = self._features[22](input=conv_4_2)
        conv_4_3 = self._features[23](input=relu_4_2)
        relu_4_3 = self._features[24](input=conv_4_3)
        conv_4_4 = self._features[25](input=relu_4_3)
        relu_4_4 = self._features[26](input=conv_4_4)
        pool_4 = self._features[27](input=relu_4_4)
        conv_5_1 = self._features[28](input=pool_4)
        relu_5_1 = self._features[29](input=conv_5_1)
        conv_5_2 = self._features[30](input=relu_5_1)
        relu_5_2 = self._features[31](input=conv_5_2)
        conv_5_3 = self._features[32](input=relu_5_2)
        relu_5_3 = self._features[33](input=conv_5_3)
        conv_5_4 = self._features[34](input=relu_5_3)
        relu_5_4 = self._features[35](input=conv_5_4)
        feature_map = {'layer_1_1': relu_1_1 if self._use_relu else conv_1_1, 'layer_1_2': relu_1_2 if self._use_relu else conv_1_2,
                       'layer_2_1': relu_2_1 if self._use_relu else conv_2_1, 'layer_2_2': relu_2_2 if self._use_relu else conv_2_2,
                       'layer_3_1': relu_3_1 if self._use_relu else conv_3_1, 'layer_3_2': relu_3_2 if self._use_relu else conv_3_2, 'layer_3_3': relu_3_3 if self._use_relu else conv_3_3, 'layer_3_4': relu_3_4 if self._use_relu else conv_3_4,
                       'layer_4_1': relu_4_1 if self._use_relu else conv_4_1, 'layer_4_2': relu_4_2 if self._use_relu else conv_4_2, 'layer_4_3': relu_4_3 if self._use_relu else conv_4_3, 'layer_4_4': relu_4_4 if self._use_relu else conv_4_4,
                       'layer_5_1': relu_5_1 if self._use_relu else conv_5_1, 'layer_5_2': relu_5_2 if self._use_relu else conv_5_2, 'layer_5_3': relu_5_3 if self._use_relu else conv_5_3, 'layer_5_4': relu_5_4 if self._use_relu else conv_5_4}
        features, features_1, features_2, features_3, features_4, features_5 = [], [], [], [], [], []
        for layer in self._layers_1:
            features_1.append(feature_map[layer])
        for layer in self._layers_2:
            features_2.append(feature_map[layer])
        for layer in self._layers_3:
            features_3.append(feature_map[layer])
        for layer in self._layers_4:
            features_4.append(feature_map[layer])
        for layer in self._layers_5:
            features_5.append(feature_map[layer])
        if features_1:
            features.append(features_1)
        if features_2:
            features.append(features_2)
        if features_3:
            features.append(features_3)
        if features_4:
            features.append(features_4)
        if features_5:
            features.append(features_5)
        return features
