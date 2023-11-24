from cv2 import applyColorMap, CHAIN_APPROX_SIMPLE, COLORMAP_JET, drawContours, findContours, imwrite, normalize, NORM_MINMAX, RETR_TREE
from dataset import TestDataset, TrainDataset
from numpy import around, array, ravel, uint8
from os.path import join
from sklearn.metrics import roc_auc_score, roc_curve
from torch import cat, flatten, no_grad, permute, sort, stack, squeeze, unsqueeze
from torch.nn import Module
from utils import convert_tensor_to_numpy, create_dir, get_device


def _binary_score(model: Module, category_path: str, image_size: tuple) -> tuple:
    scores, ground_truths = [], []
    test_dataset = TestDataset(path=category_path, size=image_size)
    with no_grad():
        for input, ground_truth, _, _ in test_dataset.dataloader:
            scores.append(convert_tensor_to_numpy(tensor=(squeeze(input=model.compute_score(input=input.to(device=get_device(module=model)))))))
            ground_truths.append(convert_tensor_to_numpy(tensor=squeeze(input=ground_truth)))
    scores = array(object=scores)
    ground_truths = array(object=ground_truths)
    ground_truths[ground_truths <= 0.5] = 0
    ground_truths[ground_truths > 0.5] = 1
    return scores, ground_truths


def compute_auc_roc(model: Module, category_path: str, image_size: tuple) -> float:
    scores, ground_truths = _binary_score(model=model, category_path=category_path, image_size=image_size)
    return roc_auc_score(y_true=ravel(a=ground_truths), y_score=ravel(a=scores))


def draw_roc_curve(model: Module, category_path: str, image_size: tuple) -> tuple:
    scores, ground_truths = _binary_score(model=model, category_path=category_path, image_size=image_size)
    return roc_curve(y_true=ravel(a=ground_truths), y_score=ravel(a=scores))


def compute_threshold(model: Module, category_path: str, image_size: tuple, expect_fprs: tuple) -> None:
    threshold_dataset = TrainDataset(path=category_path, batch_size=1, num_workers=0, size=image_size)
    with no_grad():
        scores = []
        for input in threshold_dataset.dataloader:
            scores.append(squeeze(input=model.compute_score(input=input.to(get_device(module=model)))))
        score = stack(tensors=scores, dim=0)
        print(score.shape)
        B, H, W = score.shape
        for expect_fpr in expect_fprs:
            threshold = sort(input=flatten(input=score), descending=True)[0][int(B * H * W * expect_fpr)].item()
            model.thresholds[expect_fpr] = threshold
            print('expect_fpr: {}  threshold: {}'.format(expect_fpr, threshold))


def segment(model: Module, category_path: str, save_path: str, image_size: tuple, expect_fprs: tuple) -> None:
    compute_threshold(model=model, category_path=category_path, image_size=image_size, expect_fprs=expect_fprs)
    test_dataset = TestDataset(path=category_path, size=image_size)
    with no_grad():
        for input, ground_truth, defect_category, name in test_dataset.dataloader:
            score = convert_tensor_to_numpy(tensor=unsqueeze(input=squeeze(input=model.compute_score(input=input.to(device=get_device(module=model)))), dim=2))
            input = squeeze(input=input)
            input[0] = input[0] * test_dataset.std[0] + test_dataset.mean[0]
            input[1] = input[1] * test_dataset.std[1] + test_dataset.mean[1]
            input[2] = input[2] * test_dataset.std[2] + test_dataset.mean[2]
            input = uint8(around(a=convert_tensor_to_numpy(tensor=permute(input=input, dims=(1, 2, 0))[:, :, (2, 1, 0)]) * 255))
            ground_truth = uint8(around(a=convert_tensor_to_numpy(tensor=unsqueeze(input=squeeze(input=ground_truth), dim=2))))
            path = join(save_path, defect_category[0], name[0])
            create_dir(path=path)

            heat_map = normalize(src=uint8(around(a=score * 255)), dst=None, alpha=0, beta=255, norm_type=NORM_MINMAX)
            heat_map = applyColorMap(src=heat_map, colormap=COLORMAP_JET) * 0.7 + input * 0.5
            heat_map_path = join(path, 'heat_map.png')
            imwrite(filename=heat_map_path, img=heat_map)
            print("success output：{}".format(heat_map_path))

            for expect_fpr in expect_fprs:
                segment_result = input.copy()
                segment_result[:, :, 0][score[:, :, 0] >= model.thresholds[expect_fpr]] = 124
                segment_result[:, :, 1][score[:, :, 0] >= model.thresholds[expect_fpr]] = 252
                segment_result[:, :, 2][score[:, :, 0] >= model.thresholds[expect_fpr]] = 0
                contours, _ = findContours(image=ground_truth, mode=RETR_TREE, method=CHAIN_APPROX_SIMPLE)
                drawContours(image=segment_result, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
                segment_result_path = join(path, "fpr-{}.png".format(expect_fpr))
                imwrite(filename=segment_result_path, img=segment_result)
                print("success output：{}".format(segment_result_path))
