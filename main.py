import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
import tqdm
import os
import shutil
import numpy as np
from datetime import datetime
from scipy.optimize import linear_sum_assignment


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNet100'])
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--n_estimation', type=int, default=10)
    parser.add_argument('--n_labelled', type=int, default=5)
    parser.add_argument('--n_unlabelled', type=int, default=5)
    parser.add_argument('--prop_labelled', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--p_noise', type=float, default=0.1)
    parser.add_argument('--epochs_noise', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.125)
    parser.add_argument('--epsilon', type=float, default=0.07)
    parser.add_argument('--sinkhorn_iterations', type=int, default=3)
    parser.add_argument('--record', type=bool, default=False)
    return parser.parse_args()


def get_path(args):
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    args.log_path = os.path.join(args.log_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i = 0
    while True:
        s = '' if i == 0 else '_' + str(i)
        try:
            os.mkdir(args.log_path + s)
        except FileExistsError:
            i += 1
            continue
        args.log_path = args.log_path + s
        break


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(args):
    class SiameseTransform:
        def __init__(self, transform1, transform2):
            self.transform1 = transform1
            self.transform2 = transform2

        def __call__(self, img):
            return self.transform1(img), self.transform2(img)

    normalize = {
        'CIFAR10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
        'CIFAR100': [(0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)],
        'ImageNet100': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    }

    dataset_train, dataset_test_t, dataset_test_v = None, None, None
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[args.dataset])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(32 * (8 / 7))),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[args.dataset])
        ])
        transform_train = SiameseTransform(transform_train, transform_train)

        dataset_train = getattr(datasets, args.dataset)(args.dataset_path, train=True, downoad=True, transform=transform_train)
        dataset_test_t = getattr(datasets, args.dataset)(args.dataset_path, train=True, downoad=True, transform=transform_test)
        dataset_test_v = getattr(datasets, args.dataset)(args.dataset_path, train=False, downoad=True, transform=transform_test)
    elif args.dataset == 'ImageNet100':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[args.dataset])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*normalize[args.dataset])
        ])
        transform_train = SiameseTransform(transform_train, transform_train)

        dataset_train = datasets.ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transform_train)
        dataset_test_t = datasets.ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transform_test)
        dataset_test_v = datasets.ImageFolder(os.path.join(args.dataset_path, 'val'), transform=transform_test)

    idx_labelled = [i for i, t in enumerate(dataset_train.targets) if t in range(args.n_labelled)]
    idx_labelled = np.sort(np.random.choice(idx_labelled, int(args.prop_labelled * len(idx_labelled)), replace=False))
    idx_unlabelled = np.array([i for i in range(len(dataset_train.targets)) if i not in idx_labelled])
    for i in idx_unlabelled:
        dataset_train.targets[i] -= args.n_class
        dataset_test_t.targets[i] -= args.n_class
    for i, _ in enumerate(dataset_test_v.targets):
        dataset_test_v.targets[i] -= args.n_class

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=True, pin_memory=True, drop_last=True)
    loader_test_t = torch.utils.data.DataLoader(dataset_test_t, batch_size=args.batch_size,
                                                num_workers=args.num_workers, shuffle=False, pin_memory=True)
    loader_test_v = torch.utils.data.DataLoader(dataset_test_v, batch_size=args.batch_size,
                                                num_workers=args.num_workers, shuffle=False, pin_memory=True)
    return loader_train, loader_test_t, loader_test_v


def get_backbone(args):
    if args.dataset == 'ImageNet100':
        backbone = models.resnet50()
    else:
        backbone = models.resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, dim


def train(model, loader, loader_test, optimizer, scheduler, args):
    global_progress = tqdm.tqdm(range(args.epochs), desc='Training')
    for epoch in global_progress:
        model.train()
        local_progress = tqdm.tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}')
        for (x1, x2), t in local_progress:
            model.zero_grad()
            loss = model(x1.to(args.device), x2.to(args.device), t.to(args.device))
            loss.backward()
            optimizer.step()
            local_progress.set_postfix({"loss": loss.item(), "lr": optimizer.state_dict()['param_groups'][0]['lr']})
        scheduler.step()
        if args.record:
            test(model, loader_test, args)
        model.epoch += 1


def test(model, loader, args, val=False):
    if val:
        log_path = args.log_path
    else:
        log_path = os.path.join(args.log_path, 'Infos')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    model.eval()
    result = np.zeros((args.n_class, args.n_estimation), dtype=int)
    result_sup = np.zeros((args.n_class, args.n_estimation), dtype=int)
    progress = tqdm.tqdm(loader, desc='Testing')
    with torch.no_grad():
        for data, target in progress:
            pred = torch.mm(F.normalize(model.backbone(data.to(args.device)), dim=-1),
                            F.normalize(model.prototypes.weight, dim=-1)[:args.n_estimation].t()).argmax(dim=-1)
            for i in range(len(target)):
                if target[i] < 0:
                    result[target[i] + args.n_class, pred[i]] += 1
                else:
                    result_sup[target[i], pred[i]] += 1

    row_ind, col_ind = linear_sum_assignment(result[args.n_labelled:, args.n_labelled:], maximize=True)
    row_ind = np.concatenate([np.arange(args.n_labelled), row_ind + args.n_labelled])
    col_ind = np.concatenate([np.arange(args.n_labelled), col_ind + args.n_labelled])
    acc_sup = result_sup[row_ind, col_ind].sum() / result_sup.sum()
    acc_all = result[row_ind, col_ind].sum() / result.sum()
    acc_l = result[row_ind[:args.n_labelled], col_ind[:args.n_labelled]].sum() / result[:args.n_labelled].sum()
    acc_u = result[row_ind[args.n_labelled:], col_ind[args.n_labelled:]].sum() / result[args.n_labelled:].sum()
    l2l = result[:args.n_labelled, :args.n_labelled].sum() / result[:args.n_labelled].sum()
    l2u = result[:args.n_labelled, args.n_labelled:].sum() / result[:args.n_labelled].sum()
    u2l = result[args.n_labelled:, :args.n_labelled].sum() / result[args.n_labelled:].sum()
    u2u = result[args.n_labelled:, args.n_labelled:].sum() / result[args.n_labelled:].sum()

    print('Dataset: {}-{}/{}'.format(args.dataset, args.n_labelled, args.n_unlabelled))
    print('Accuracy: Sup={:.2f}%'.format(100 * acc_sup))
    print('Accuracy: All={:.2f}%  Old={:.2f}%  New={:.2f}%'.format(100 * acc_all, 100 * acc_l, 100 * acc_u))
    print('Accuracy: L2L={:.2f}%  L2U={:.2f}%'.format(100 * l2l, 100 * l2u))
    print('Accuracy: U2L={:.2f}%  U2U={:.2f}%'.format(100 * u2l, 100 * u2u))
    if args.dataset == 'CIFAR10' and args.n_class == args.n_estimation:
        labels = ['Plane', 'Car ', 'Bird', 'Cat ', 'Deer', 'Dog ', 'Frog', 'Horse', 'Ship', 'Truck']
        print('Tgt\\Prd', *labels, 'RECALL(%)', sep='\t')
        for i, t in enumerate(result):
            print(labels[i], end='\t')
            for j, p in enumerate(t):
                print('{:<5d}'.format(p), '*' if j == col_ind[i] else ' ', end='\t', sep='')
            print('{:.2f}%'.format(100 * t[col_ind[i]] / t.sum()))
        print('SUM ', end='\t')
        for i in result.sum(axis=0):
            print('{:<5d}'.format(i), end='\t')
        print('{:<5d}'.format(result.sum()))
        print('PRE(%)', end='\t')
        for i in range(10):
            p = result[np.argwhere(col_ind == i).item(), i] / result.sum(axis=0)[i] if result.sum(axis=0)[i] else 0
            print('{:.2f}%'.format(100 * p), end='\t')
        print()

    file_name = 'Epoch,L2L,L2U,U2L,U2U.csv'
    with open(os.path.join(log_path, file_name), 'a') as f:
        print('{},{:.2f},{:.2f},{:.2f},{:.2f}'.format(model.epoch, 100 * l2l, 100 * l2u, 100 * u2l, 100 * u2u), file=f)

    file_name = 'Epoch{}_{:.2f}%_{:.2f}%_{:.2f}%.txt'.format(model.epoch, 100 * acc_all, 100 * acc_l, 100 * acc_u)
    with open(os.path.join(log_path, file_name), 'w') as f:
        print('Dataset: {}-{}/{}'.format(args.dataset, args.n_labelled, args.n_unlabelled), file=f)
        print('Accuracy: Sup={:.2f}%'.format(100 * acc_sup), file=f)
        print('Accuracy: All={:.2f}%  Old={:.2f}%  New={:.2f}%'.format(100 * acc_all, 100 * acc_l, 100 * acc_u), file=f)
        print('Accuracy: L2L={:.2f}%  L2U={:.2f}%'.format(100 * l2l, 100 * l2u), file=f)
        print('Accuracy: U2L={:.2f}%  U2U={:.2f}%'.format(100 * u2l, 100 * u2u), file=f)
        print('Args:', file=f)
        for k, v in vars(args).items():
            print('   ', k, '=', v, file=f)
        print(file=f)
        if args.dataset == 'CIFAR10' and args.n_class == args.n_estimation:
            labels = ['Plane', 'Car ', 'Bird', 'Cat ', 'Deer', 'Dog ', 'Frog', 'Horse', 'Ship', 'Truck']
            print('Tgt\\Prd', *labels, 'RECALL(%)', sep='\t', file=f)
            for i, t in enumerate(result):
                print(labels[i], end='\t', file=f)
                for j, p in enumerate(t):
                    print('{:<5d}'.format(p), '*' if j == col_ind[i] else ' ', end='\t', sep='', file=f)
                print('{:.2f}%'.format(100 * t[col_ind[i]] / t.sum()), file=f)
            print('SUM ', end='\t', file=f)
            for i in result.sum(axis=0):
                print('{:<5d}'.format(i), end='\t', file=f)
            print('{:<5d}'.format(result.sum()), file=f)
            print('PRE(%)', end='\t', file=f)
            for i in range(10):
                p = result[np.argwhere(col_ind == i).item(), i] / result.sum(axis=0)[i] if result.sum(axis=0)[i] else 0
                print('{:.2f}%'.format(100 * p), end='\t', file=f)


def main():
    args = get_args()
    get_path(args)
    set_seeds(args.seed)
    loader_train, loader_test_t, loader_test_v = get_data(args)
    model = Model(*get_backbone(args), args).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    train(model, loader_train, loader_test_t, optimizer, scheduler, args)
    torch.save(model.state_dict(), os.path.join(args.log_path, "model.pt"))
    test(model, loader_test_v, args, True)
    shutil.copyfile(__file__, os.path.join(args.log_path, '{}-{}.py'.format(args.dataset, args.n_unlabelled)))


class Model(nn.Module):
    def __init__(self, backbone, dim, args):
        super().__init__()
        self.backbone = backbone
        self.n_class = args.n_class
        self.n_estimation = args.n_estimation
        self.prototypes = nn.Linear(in_features=dim, out_features=self.n_class * 2, bias=False)
        self.temperature = args.temperature
        self.epsilon = args.epsilon
        self.sinkhorn_iterations = args.sinkhorn_iterations
        self.epoch = 0

    def forward(self, x1, x2, t):
        idx_m = torch.randperm(len(t), device=t.device)
        x1_m, x2_m = (x1 + x2[idx_m]) / 2, (x2 + x1[idx_m]) / 2
        z1, z2 = F.normalize(self.backbone(x1), dim=-1), F.normalize(self.backbone(x2), dim=-1)
        z1_m, z2_m = F.normalize(self.backbone(x1_m), dim=-1), F.normalize(self.backbone(x2_m), dim=-1)
        c = F.normalize(self.prototypes.weight, dim=-1)[:self.n_estimation]
        p1 = F.softmax(torch.mm(z1, c.t()) / self.temperature, dim=-1)
        p2 = F.softmax(torch.mm(z2, c.t()) / self.temperature, dim=-1)
        p1_m = F.softmax(torch.mm(z1_m, c.t()) / self.temperature, dim=-1)
        p2_m = F.softmax(torch.mm(z2_m, c.t()) / self.temperature, dim=-1)
        q1 = self.sinkhorn(torch.exp(torch.mm(z1, c.t()) / self.epsilon), t).detach()
        q2 = self.sinkhorn(torch.exp(torch.mm(z2, c.t()) / self.epsilon), t).detach()
        q1_m, q2_m = (q1 + q2[idx_m]) / 2, (q2 + q1[idx_m]) / 2
        loss = torch.mean(-torch.sum(q2 * torch.log(p1), dim=-1)) + torch.mean(-torch.sum(q1 * torch.log(p2), dim=-1))
        loss_m = torch.mean(-torch.sum(q2_m * torch.log(p1_m), dim=-1)) + \
                 torch.mean(-torch.sum(q1_m * torch.log(p2_m), dim=-1))
        return (loss + loss_m) / 2

    def sinkhorn(self, q, t):
        q = F.normalize(q, dim=1, p=1)
        for _ in range(self.sinkhorn_iterations):
            q = F.normalize(q, dim=0, p=1)  # normalize each column: total weight per prototype must be 1
            q = F.normalize(q, dim=1, p=1)  # normalize each row: total weight per sample must be 1
        m = torch.zeros_like(q).scatter_(1, torch.unsqueeze(torch.maximum(t, torch.zeros_like(t)), dim=1), 1)
        q = torch.where(torch.unsqueeze(t, dim=1).ge(0), m, q)
        return q


if __name__ == '__main__':
    main()
