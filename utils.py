import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cvxpy as cp
import warnings

from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
from models import ResNet18, ResNet50, VGG, WideResNet

warnings.filterwarnings("ignore")
os.environ['MOSEKLM_LICENSE_FILE'] = "mosek.lic"


def infer_exp_name_clean(r_choice, train_loss, eps, lr, epochs, arch, seed=0, schedule=None):
    if schedule:
        exp_name = 'r{}-{}-lr{}-e{}-a{}-seed{}-lr_decay'.format(
            r_choice,
            train_loss if train_loss == 'ST' else '{}{:.1f}'.format(train_loss, eps * 255),
            lr,
            epochs,
            arch,
            seed)
    else:
        exp_name = 'r{}-{}-lr{}-e{}-a{}-seed{}'.format(
            r_choice,
            train_loss if train_loss == 'ST' else '{}{:.1f}'.format(train_loss, eps * 255),
            lr,
            epochs,
            arch,
            seed)
    return exp_name


def infer_exp_name_hr(r_choice, train_loss, eps, epochs, arch, seed=0, schedule=None):
    if schedule:
        exp_name = 'r{}-{}-e{}-a{}-seed{}-lr_decay'.format(
            r_choice,
            train_loss if train_loss == 'ST' else '{}{:.1f}'.format(train_loss, eps * 255),
            epochs,
            arch,
            seed)
    else:
        exp_name = 'r{}-{}-e{}-a{}-seed{}'.format(
            r_choice,
            train_loss if train_loss == 'ST' else '{}{:.1f}'.format(train_loss, eps * 255),
            epochs,
            arch,
            seed)
    return exp_name


def infer_exp_name_wdro(r_choice, lr, eps, epochs, arch, seed=0, schedule=None):
    if schedule:
        exp_name = 'r{}-{}-lr{}-e{}-a{}-seed{}-lr_decay'.format(
            r_choice,
            '{}{:.1f}'.format("UDR", eps * 255),
            lr,
            epochs,
            arch,
            seed)
    else:
        exp_name = 'r{}-{}-lr{}-e{}-a{}-seed{}'.format(
            r_choice,
            '{}{:.1f}'.format("UDR", eps * 255),
            lr,
            epochs,
            arch,
            seed)
    return exp_name


def make_and_restore_model(arch, dataset='CIFAR10', resume_path=None):
    if dataset == "CIFAR10":
        if arch == 'ResNet18':
            model = ResNet18()
        elif arch == 'VGG16':
            model = VGG('VGG16')
        elif arch == "ResNet50":
            model = ResNet50()
        elif arch == 'WRN28-10':
            model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model = InputNormalize(model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616))
    elif dataset == "CIFAR100":
        if arch == 'ResNet18':
            model = ResNet18(num_classes=100)
        elif arch == "ResNet50":
            model = ResNet50(num_classes=100)
        elif arch == 'VGG16':
            model = VGG('VGG16', num_classes=100)
        elif arch == 'WRN28-10':
            model = WideResNet(depth=28, num_classes=100, widen_factor=10)
        model = InputNormalize(model, new_mean=(0.5071, 0.4865, 0.4409), new_std=(0.2673, 0.2564, 0.2762))
    elif dataset == "SVHN":
        if arch == 'ResNet18':
            model = ResNet18(num_classes=10)
        elif arch == 'VGG16':
            model = VGG('VGG16', num_classes=10)
        elif arch == 'WRN28-10':
            model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model = InputNormalize(model, new_mean=(0.4377, 0.4438, 0.4728), new_std=(0.1980, 0.2010, 0.1970))
    elif dataset == "Tiny-Imagenet":
        if arch == 'ResNet18':
            model = ResNet18(num_classes=200)
        elif arch == 'VGG16':
            model = VGG('VGG16', num_classes=200)
        elif arch == 'WRN28-10':
            model = WideResNet(depth=28, num_classes=200, widen_factor=10)
        model = InputNormalize(model, new_mean=(0.4802, 0.4481, 0.3975), new_std=(0.2770, 0.2691, 0.2821))
    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        # info_keys = ['epoch', 'train_acc', 'cln_val_acc', 'cln_test_acc', 'adv_val_acc', 'adv_test_acc']
        info = {checkpoint['epoch']}
        pprint(info)
        resume_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

        model = model.cuda()
        return model, resume_epoch
    else:
        model = model.cuda()
        return model


class InputNormalize(nn.Module):
    def __init__(self, model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616)):
        super(InputNormalize, self).__init__()
        new_mean = torch.tensor(new_mean)[..., None, None]
        new_std = torch.tensor(new_std)[..., None, None]
        self.register_buffer('new_mean', new_mean)
        self.register_buffer('new_std', new_std)
        self.model = model

    def __call__(self, x):
        x = (x - self.new_mean) / self.new_std
        return self.model(x)


def set_seed(seed):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)


def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k
        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes)
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)
        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def get_mean_std(loader):
    imgs = None
    for batch in loader:
        image_batch = batch[0]
        if imgs is None:
            imgs = image_batch.cpu()
        else:
            imgs = torch.cat([imgs, image_batch.cpu()], dim=0)
    imgs = imgs.numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()
    print(mean_r, mean_g, mean_b)
    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()
    print(std_r, std_g, std_b)


class DPPOptimizer:
    def __init__(self, r, num_eps):
        self.r = r
        self.num_eps = num_eps
        self.optimizers = {}

    def _create_optimizer(self, n):
        a = cp.Variable(n)
        loss = cp.Parameter(n)
        initial_weights = np.ones(n) / n

        objective = cp.Maximize(cp.sum(cp.multiply(a, loss)))

        # Exponential cone constraints
        t = cp.Variable(name="t", shape=n)
        exc_constraints = [cp.constraints.exponential.ExpCone(-1 * t, initial_weights, a)]
        extra_constraints = [cp.sum(t) <= self.r]
        simplex_constraints = [cp.sum(a) == 1]
        complete_constraints = simplex_constraints + exc_constraints + extra_constraints
        # complete_constraints = [
        #     cp.sum(a) == 1,
        #     # a >= 0,
        #     cp.kl_div(initial_weights, a) <= self.r
        # ]
        prob = cp.Problem(objective, complete_constraints)
        return {'a': a, 'loss': loss, 'prob': prob}

    def optimize(self, loss_values):
        n = len(loss_values)
        if n not in self.optimizers:
            self.optimizers[n] = self._create_optimizer(n)

        opt = self.optimizers[n]
        try:
            opt['prob'].solve(solver=cp.ECOS)
        except:
            opt['loss'].value = loss_values + self.num_eps  # 添加少量噪声
            opt['prob'].solve(solver=cp.MOSEK)

        return opt['a'].value

