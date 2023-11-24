from os import listdir
from os.path import join
from PIL.Image import open
from torch import repeat_interleave, Tensor, zeros
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from utils import get_file_name


class TrainDataset(Dataset):
    def __init__(self, path: str, batch_size: int, num_workers: int, size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(TrainDataset, self).__init__()
        self._image_paths = []
        path = join(path, 'train', 'good')
        for image_name in listdir(path=path):
            self._image_paths.append(join(path, image_name))
        self._resize = Resize(size=size, interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = Normalize(mean=mean, std=std)
        self.size = size
        self.mean = mean
        self.std = std
        self.dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    def __getitem__(self, index: int) -> Tensor:  # (B, C, H, W)
        image = self._tensor_transform(pic=self._resize(img=open(fp=self._image_paths[index])))
        if image.shape[0] == 1:
            image = repeat_interleave(input=image, repeats=3, dim=0)
        return self._normalize_transform(tensor=image)

    def __len__(self) -> int:
        return len(self._image_paths)


class TestDataset(Dataset):
    def __init__(self, path: str, size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(TestDataset, self).__init__()
        self._test_image_paths = []
        self._ground_truth_image_paths = []
        self._defect_categories = []
        test_path = join(path, 'test')
        ground_truth_path = join(path, 'ground_truth')
        for test_category_name in listdir(path=test_path):
            test_category_path = join(test_path, test_category_name)
            if test_category_name == 'good':
                for test_image_name in listdir(path=test_category_path):
                    self._test_image_paths.append(join(test_category_path, test_image_name))
                    self._ground_truth_image_paths.append('')
                    self._defect_categories.append('good')
            else:
                ground_truth_category_path = join(ground_truth_path, test_category_name)
                for test_image_name, ground_truth_image_name in zip(listdir(path=test_category_path), listdir(path=ground_truth_category_path)):
                    self._test_image_paths.append(join(test_category_path, test_image_name))
                    self._ground_truth_image_paths.append(join(ground_truth_category_path, ground_truth_image_name))
                    self._defect_categories.append(test_category_name)
        self._resize = Resize(size=size, interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = Normalize(mean=mean, std=std)
        self.size = size
        self.mean = mean
        self.std = std
        self.dataloader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=1)

    def __getitem__(self, index: int) -> tuple:  # (B, C, H, W), (B, H, W)
        test_image = self._tensor_transform(pic=self._resize(img=open(fp=self._test_image_paths[index])))
        if test_image.shape[0] == 1:
            test_image = repeat_interleave(input=test_image, repeats=3, dim=0)
        ground_truth_image = zeros(size=self.size) if self._ground_truth_image_paths[index] == '' else self._tensor_transform(pic=self._resize(img=open(fp=self._ground_truth_image_paths[index])))
        return self._normalize_transform(tensor=test_image), ground_truth_image, self._defect_categories[index], get_file_name(path=self._test_image_paths[index])

    def __len__(self) -> int:
        return len(self._test_image_paths)
