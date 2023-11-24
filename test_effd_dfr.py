from argparse import ArgumentParser
from model.effd_dfr import EffdDFR
from model.extractor import Vgg19Extractor
from os.path import join
from torch import device, load
from evaluate import segment
from utils import create_dir


if __name__ == '__main__':
    argumentParser = ArgumentParser()
    argumentParser.add_argument('--device', type=int, default=0)

    argumentParser.add_argument('--num_workers', type=int, default=15)
    argumentParser.add_argument("--image_size", nargs='+', type=int, default=(256, 256))



    argumentParser.add_argument('--pool', type=str, choices=('avgpool', 'maxpool'), default='avgpool')
    argumentParser.add_argument('--padding_mode', type=str, choices=('zeros', 'reflect', 'replicate'), default='reflect')
    argumentParser.add_argument('--use_relu', type=int, choices=(0, 1), default=0)
    argumentParser.add_argument("--layers", nargs='+', type=str, default=('layer_2_1', 'layer_2_2', 'layer_3_1', 'layer_3_2', 'layer_3_3', 'layer_3_4', 'layer_4_1', 'layer_4_2', 'layer_4_3', 'layer_4_4'))

    argumentParser.add_argument("--expect_fprs", nargs='+', type=float, default=(0.0005,))

    argumentParser.add_argument("--category", type=str, required=True)
    argumentParser.add_argument("--data_path", type=str, default='../dfr/data')
    argumentParser.add_argument("--save_path", type=str, default='result')
    argumentParser.add_argument('--load_iaffs_path', type=str, required=True)
    argumentParser.add_argument('--load_cae_path', type=str, required=True)
    args = argumentParser.parse_args()

    device = device('cuda:{}'.format(args.device))

    print("--------------------------category：{}--------------------------".format(args.category))
    category_path = join(args.data_path, args.category)
    save_path = join(args.save_path, args.category)
    create_dir(path=save_path)

    extractor = Vgg19Extractor(pool=args.pool, padding_mode=args.padding_mode, use_relu=bool(args.use_relu), layers=args.layers)
    model = EffdDFR(extractor=extractor, iaffs=load(f=args.load_iaffs_path), cae=load(f=args.load_cae_path), image_size=args.image_size, eta=args.eta).to(device=device).eval()

    segment(model=model, category_path=category_path, save_path=save_path, image_size=args.image_size, expect_fprs=args.expect_fprs)

    print("testing completed")
