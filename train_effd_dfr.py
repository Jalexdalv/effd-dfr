from argparse import ArgumentParser
from dataset import TrainDataset
from evaluate import compute_auc_roc
from model.cae import CAE
from model.effd_dfr import EffdDFR
from model.extractor import Vgg19Extractor
from model.iaff import IAFF
from os.path import exists, join
from torch import device, load, no_grad, save
from torch.nn import ModuleList
from torch.optim import AdamW
from tqdm import tqdm
from utils import create_dir, load_list, save_list


if __name__ == '__main__':
    argumentParser = ArgumentParser()
    argumentParser.add_argument('--device', type=int, default=0)

    argumentParser.add_argument('--iaff_batch_size', type=int, default=4)
    argumentParser.add_argument('--iaff_num_workers', type=int, default=15)
    argumentParser.add_argument('--iaff_num_epochs', type=int, default=100)
    argumentParser.add_argument('--iaff_lr', type=float, default=1e-3)
    argumentParser.add_argument('--iaff_weight_decay', type=float, default=1e-3)

    argumentParser.add_argument('--cae_batch_size', type=int, default=8)
    argumentParser.add_argument('--cae_num_workers', type=int, default=15)
    argumentParser.add_argument('--cae_num_epochs', type=int, default=30)
    argumentParser.add_argument('--cae_lr', type=float, default=1e-3)
    argumentParser.add_argument('--cae_weight_decay', type=float, default=2.)

    argumentParser.add_argument('--image_size', nargs='+', type=int, default=(256, 256))

    argumentParser.add_argument('--alpha', type=int, default=3)
    argumentParser.add_argument('--betas', nargs='+', type=int, default=(2, 2, 2))
    argumentParser.add_argument('--gamma', type=int, default=4)
    argumentParser.add_argument('--eta', type=int, default=8)
    argumentParser.add_argument('--pool', type=str, choices=('avgpool', 'maxpool'), default='avgpool')
    argumentParser.add_argument('--padding_mode', type=str, choices=('zeros', 'reflect', 'replicate'), default='reflect')
    argumentParser.add_argument('--use_relu', type=int, choices=(0, 1), default=0)
    argumentParser.add_argument('--layers', nargs='+', type=str, default=('layer_2_1', 'layer_2_2', 'layer_3_1', 'layer_3_2', 'layer_3_3', 'layer_3_4', 'layer_4_1', 'layer_4_2', 'layer_4_3', 'layer_4_4'))

    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('wood', 'tile', 'cable', 'metal_nut', 'transistor'))
    argumentParser.add_argument('--categories', nargs='+', type=str, default=('wood', 'tile', 'cable', 'metal_nut', 'transistor', 'bottle', 'capsule', 'leather', 'hazelnut', 'carpet', 'pill', 'screw', 'toothbrush', 'grid', 'zipper'))
    argumentParser.add_argument('--data_path', type=str, default='../dfr/data')
    argumentParser.add_argument('--save_path', type=str, default='pretrain')
    argumentParser.add_argument('--evaluate_interval', type=int, default=1)
    argumentParser.add_argument('--save_interval', type=int, default=1)
    args = argumentParser.parse_args()

    device = device('cuda:{}'.format(args.device))

    for category in args.categories:
        print('--------------------------category：{}--------------------------'.format(category))
        category_path = join(args.data_path, category)
        save_path = join(args.save_path, category)
        create_dir(path=save_path)
        distribution_path = join(save_path, 'distributions-{}x{}-{}-{}-{}-[{}].npy'.format(args.image_size[0], args.image_size[1], args.eta, args.pool, args.padding_mode, '-'.join(args.layers)))
        iaff_path = join(save_path, 'iaffs-{}x{}-{}-{}-{}-{}-{}-{}-{}-{}-[{}].pth'.format(args.image_size[0], args.image_size[1], args.iaff_batch_size, args.iaff_num_epochs, args.iaff_lr, args.iaff_weight_decay, args.gamma, args.eta, args.pool, args.padding_mode, '-'.join(args.layers)))
        enable_effd = not exists(path=distribution_path)
        train_iaff = not exists(path=iaff_path)

        extractor = Vgg19Extractor(pool=args.pool, padding_mode=args.padding_mode, use_relu=bool(args.use_relu), layers=args.layers).to(device=device).eval()
        iaffs = ModuleList([ModuleList([IAFF(num_in_channels=channels[0], gamma=args.gamma).to(device=device) for _ in range(len(channels) - 1)]) for channels in extractor.channels]) if train_iaff else load(f=iaff_path).to(device=device).eval()
        cae = CAE(num_in_channels=sum([channels[0] for channels in extractor.channels]), alpha=args.alpha, betas=args.betas).to(device=device)
        model = EffdDFR(extractor=extractor, iaffs=iaffs, cae=cae, image_size=args.image_size, eta=args.eta).to(device=device)

        if train_iaff:
            train_dataset = TrainDataset(path=category_path, batch_size=args.iaff_batch_size, num_workers=args.iaff_num_workers, size=args.image_size)
            if enable_effd:
                print('------------estimating and fusing feature distribution------------')
                with no_grad():
                    print('------extracting------')
                    ms_features = [[[] for _ in channels] for channels in extractor.channels]
                    with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                        for batch in batches:
                            for ms_index, ml_features in enumerate(extractor(input=batch.to(device=device))):
                                for ml_index, feature in enumerate(ml_features):
                                    ms_features[ms_index][ml_index].append(feature)
                    print('extracting completed')
                    print('------estimating and fusing------')
                    model.effd(ms_features=ms_features)
                    print('------saving------')
                    save_list(data=model.distributions, path=distribution_path)
                    print('saving completed')
                    print('estimating and fusing completed')
            else:
                model.distributions = load_list(path=distribution_path)
                print('distributions loaded')
            optimizer = AdamW(params=iaffs.parameters(), lr=args.iaff_lr, weight_decay=args.iaff_weight_decay)
            print('------------training iaffs------------')
            for index in range(1, args.iaff_num_epochs + 1):
                print('epoch：{}'.format(index))
                loss_sum, num_samples = 0, 0
                with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                    for batch in batches:
                        loss = model.compute_distribution_loss(input=batch.to(device=device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                        num_samples += batch.shape[0]
                print('loss：{}'.format(loss_sum / num_samples))
            print('------saving------')
            save(obj=iaffs, f=iaff_path)
            print('saving completed')
            iaffs.eval()
            print('training iaffs completed')
        else:
            print('iaffs loaded')

        train_dataset = TrainDataset(path=category_path, batch_size=args.cae_batch_size, num_workers=args.cae_num_workers, size=args.image_size)
        optimizer = AdamW(params=cae.parameters(), lr=args.cae_lr, weight_decay=args.cae_weight_decay)
        print('------------training cae------------')
        for index in range(1, args.cae_num_epochs + 1):
            print('epoch：{}'.format(index))
            loss_sum, num_samples, auc_roc = 0, 0, 0
            with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                for batch in batches:
                    loss = model.compute_reconstruction_loss(input=batch.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    num_samples += batch.shape[0]
            print('loss：{}'.format(loss_sum / num_samples))

            if index % args.evaluate_interval == 0:
                print('------evaluating------')
                cae.eval()
                auc_roc = compute_auc_roc(model=model, category_path=category_path, image_size=args.image_size)
                cae.train()
                print('auc-roc：{}'.format(auc_roc))
                print('evaluating completed')

            if index % args.save_interval == 0:
                print('------saving------')
                save(obj=cae, f=join(save_path, 'effd-dfr-cae-{}x{}-{}-{}:{}-{}-{}-{}-{}-{}-{}-{}-[{}].pth'.format(args.image_size[0], args.image_size[1], args.cae_batch_size, args.cae_num_epochs, index, auc_roc, args.cae_lr, args.cae_weight_decay, args.gamma, args.eta, args.pool, args.padding_mode, '-'.join(args.layers))))
                print('saving completed')
        print('training cae completed')
