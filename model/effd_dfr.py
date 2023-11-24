from kornia.filters import gaussian_blur2d
from model.extractor import Extractor
from sklearn.covariance import LedoitWolf
from torch import abs, cat, flatten, mean, mm, permute, pow, repeat_interleave, reshape, sqrt, Tensor, unbind, zeros
from torch.functional import F
from torch.linalg import cholesky, inv
from torch.nn import Module, ModuleList
from utils import convert_numpy_to_tensor, convert_tensor_to_numpy


class EffdDFR(Module):
    def __init__(self, extractor: Extractor, iaffs: ModuleList, cae: Module, image_size: tuple, eta: int) -> None:
        super(EffdDFR, self).__init__()
        self._extractor = extractor
        self._iaffs = iaffs
        self._cae = cae
        self._image_size = image_size
        self._eta = eta
        self.thresholds = {}
        self.distributions = [{} for _ in extractor.channels]

    def effd(self, ms_features: list) -> None:
        for ms_index, ml_features in enumerate(ms_features):
            for ml_index, features in enumerate(ml_features):
                feature = cat(tensors=[F.adaptive_avg_pool2d(input=F.interpolate(input=feature, size=self._image_size, mode='bilinear', align_corners=True), output_size=(self._image_size[0] // self._eta, self._image_size[1] // self._eta)) for feature in features], dim=0)  # (B, C, H/η, W/η)
                B, C, _, _ = feature.shape
                feature = reshape(input=feature, shape=(B, C, -1))  # (B, C, H/η, W/η)
                feature = cat(tensors=unbind(input=permute(input=feature, dims=(0, 2, 1)), dim=0), dim=0)  # (B * H/η * W/η, C)
                mu = mean(input=feature.T, dim=1, keepdim=True)  # (C, 1)
                sigma = cholesky(input=convert_numpy_to_tensor(array=LedoitWolf().fit(X=convert_tensor_to_numpy(tensor=feature)).covariance_).to(device=feature.device))  # (C, C)
                P = mm(input=self.distributions[ms_index]['sigma'], mat2=inv(A=self.distributions[ms_index]['sigma'] + sigma)) if 'sigma' in self.distributions[ms_index] else None
                self.distributions[ms_index]['mu'] = self.distributions[ms_index]['mu'] + mm(input=P, mat2=mu - self.distributions[ms_index]['mu']) if 'mu' in self.distributions[ms_index] else mu
                self.distributions[ms_index]['sigma'] = self.distributions[ms_index]['sigma'] - mm(input=P, mat2=self.distributions[ms_index]['sigma']) if 'sigma' in self.distributions[ms_index] else sigma

    def compute_reconstruction_loss(self, input: Tensor) -> Tensor:
        feature, output = self(input=input)
        return F.mse_loss(input=feature, target=output)

    def compute_distribution_loss(self, input: Tensor) -> Tensor:
        loss = zeros(size=(1,)).to(device=input.device)
        for index, feature in enumerate(self._fuse_feature(input=input)):
            B, C, H, W = feature.shape
            feature = reshape(input=feature, shape=(B, C, -1))  # (B, C, H/η * W/η)
            feature = cat(tensors=unbind(input=permute(input=feature, dims=(0, 2, 1)), dim=0), dim=0)  # (B * H/η * W/η, C)
            residual = feature.T - repeat_interleave(input=self.distributions[index]['mu'], repeats=B * H * W, dim=1)  # (C, B * H/η * W/η)
            mahalanobis_distance = sqrt(input=abs(input=mm(input=mm(input=residual.T, mat2=inv(A=self.distributions[index]['sigma'])), mat2=residual)) + 1e-8)
            loss += mean(input=flatten(input=mahalanobis_distance), dim=0)
        return loss / len(self._extractor.channels)

    def compute_score(self, input: Tensor) -> Tensor:  # (B, H, W)
        feature, output = self(input=input)
        return gaussian_blur2d(input=mean(input=F.interpolate(input=pow(input=feature - output, exponent=2), size=self._image_size, mode='bilinear', align_corners=True), dim=1, keepdim=True), kernel_size=3, sigma=(3, 3))

    def forward(self, input: Tensor) -> tuple:
        feature = cat(tensors=[feature for feature in self._fuse_feature(input=input)], dim=1)
        return feature, self._cae(input=feature)

    def _fuse_feature(self, input: Tensor) -> Tensor:
        for ml_iaffs, ml_features in zip(self._iaffs, self._extractor(input=input)):
            ml_features = [F.adaptive_avg_pool2d(input=F.interpolate(input=feature, size=self._image_size, mode='bilinear', align_corners=True), output_size=(self._image_size[0] // self._eta, self._image_size[1] // self._eta)) for feature in ml_features]
            feature = ml_features[0]
            for index, iaff in enumerate(ml_iaffs):
                feature = iaff(shallow_feature=feature, deep_feature=ml_features[index + 1])
            yield feature
