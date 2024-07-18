import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from utils import AverageMeter, accuracy_top1, DPPOptimizer
from attacks.natural import natural_attack
from attacks.adv import adv_attack, batch_adv_attack
from attacks.udr import attack_udr


def adv_loss(args, model, x, y, lamda, dpp_optim):
    delta = attack_udr(model, x, y, args, lamda)
    x_adv = torch.clamp(x+delta, 0, 1)

    logits_adv = model(x_adv)
    if args.r_choice == 0:
        loss = nn.CrossEntropyLoss()(logits_adv, y)
    else:
        loss = nn.CrossEntropyLoss(reduction="none")(logits_adv, y)
        loss_list = np.array(loss.cpu().detach().numpy())

        max_weights = torch.from_numpy(dpp_optim.optimize(loss_list)).float().to(x.device)
        loss = (max_weights * loss).sum()

    return loss, logits_adv, delta


def trades_loss(args, model, x, y, lamda, dpp_optim, beta=6.0):
    delta = attack_udr(model, x, y, args, lamda)
    x_adv = torch.clamp(x + delta, 0, 1)

    logits = model(torch.cat((x, x_adv), dim=0))
    logits_cln, logits_adv = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
    if args.r_choice > 0:
        kl = nn.KLDivLoss(reduction='none')
        loss_rob = kl(F.log_softmax(logits_adv, dim=1)+1e-8, F.softmax(logits_cln, dim=1)+1e-8).sum(dim=1)
        loss_nat = nn.CrossEntropyLoss(reduction='none')(logits_cln, y)
        loss = loss_nat + beta * loss_rob
        loss_list = np.array(loss.cpu().detach().numpy())
        max_weights = torch.from_numpy(dpp_optim.optimize(loss_list)).float().to(x.device)
        loss = (max_weights * loss).sum()
    else:
        kl = nn.KLDivLoss(reduction='batchmean')
        loss_rob = kl(F.log_softmax(logits_adv, dim=1)+1e-8, F.softmax(logits_cln, dim=1)+1e-8)
        loss_nat = nn.CrossEntropyLoss()(logits_cln, y)
        loss = loss_nat + beta * loss_rob

    return loss, logits_cln, delta


LOSS_FUNC = {
    'AT': adv_loss,
    'Trades': trades_loss,
}


def mynorm(x, order):
    """
    Custom norm, given x is 2D tensor [b, d]. always calculate norm on the dim=1
        L1(x) = 1/d * sum(abs(x_i))
        L2(x) = sqrt(1/d * sum(square(x)))
        Linf(x) = max(abs(x_i))
    """
    x = torch.reshape(x, [x.shape[0], -1])
    b, d = x.shape
    if order == 1:
        return 1./d * torch.sum(torch.abs(x), dim=1) # [b,]
    elif order == 2:
        return torch.sqrt(1./d * torch.sum(torch.square(x), dim=1)) # [b,]
    elif order == np.inf:
        return torch.max(torch.abs(x), dim=1)[0] # [b,]
    else:
        raise ValueError


def train(args, model, optimizer, loader, writer, epoch, lamda, dis_cum):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95)
    dpp_optimizer = DPPOptimizer(r=args.r_choice, num_eps=args.numerical_eps)
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        loss, logits, delta = LOSS_FUNC[args.udr_type](args, model, inp, target, lamda, dpp_optimizer)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)

        # Updating Lambda - Refer to Eq 19 in our paper
        # cummulatively collect the difference and
        # manually update lamda after a period (e.g., after 10 iterations) (SGD)
        dis = torch.mean(mynorm(delta, order=1), dim=0)

        dis_cum.append(dis)
        if i % args.lamda_period == 0:
            lamda = lamda - args.lamda_lr * (args.eps - sum(dis_cum) / len(dis_cum))
            dis_cum = []

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg


def train_model(args, model, optimizer, train_loader, test_loader, writer, resume=0, schedule=None):
    # Init lamda
    dis_cum = []
    lamda = torch.tensor(args.lamda_init, requires_grad=False)
    rob_acc = 0
    for epoch in range(resume+1, args.epochs+1):
        train(args, model, optimizer, train_loader, writer, epoch, lamda, dis_cum)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            # nat_clean_train_loss, nat_clean_train_acc = natural_attack(
            #     args, model, train_loader, writer, epoch, 'clean_train')
            nat_clean_test_loss, nat_clean_test_acc = natural_attack(
                args, model, test_loader, writer, epoch, 'clean_test')

            # adv_clean_train_loss, adv_clean_train_acc, _ = adv_attack(args, model, train_loader,
            #                                                           writer, epoch, 'clean_train')
            adv_clean_test_loss, adv_clean_test_acc, _ = adv_attack(args, model, test_loader, writer,
                                                                    epoch, 'clean_test')

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': -1,
                'train_loss': -1,
                # 'nat_clean_train_acc': nat_clean_train_acc,
                'nat_clean_test_acc': nat_clean_test_acc,
                # 'adv_clean_train_acc': adv_clean_train_acc,
                'adv_clean_test_acc': adv_clean_test_acc,
            }
            if adv_clean_test_acc > rob_acc:
                rob_acc = adv_clean_test_acc
                torch.save(checkpoint, args.model_path_best)
            torch.save(checkpoint, args.model_path_last)
        if schedule:
            schedule.step()
    print("The best test adv acc:{:.4f}".format(rob_acc))
    return model


def eval_model(args, model, loader):
    model.eval()
    args.eps = args.eps

    keys, values = [], []
    keys.append('Model')
    values.append(args.model_path_last)

    # Natural
    _, acc = natural_attack(args, model, loader)
    keys.append("nat")
    values.append(acc)

    # FGSM
    args.num_steps = 1
    args.step_size = args.eps
    args.random_restarts = 0
    _, acc, name = adv_attack(args, model, loader)
    keys.append('FGSM')
    values.append(acc)

    # PGD-10
    args.num_steps = 10
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # PGD-20
    args.num_steps = 20
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # PGD-200
    args.num_steps = 200
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, loader)
    keys.append(name)
    values.append(acc)

    # AutoAttack
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.constraint, eps=args.eps, version='standard')
    x_test = torch.cat([x for (x, y) in loader])
    y_test = torch.cat([y for (x, y) in loader])
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    auto_acc = adversary.clean_accuracy(x_adv, y_test, bs=args.batch_size) * 100
    keys.append('AotuAttack')
    values.append(auto_acc)

    # Save results
    import csv
    csv_fn = '{}.csv'.format(args.model_path_last)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)

    print('=> csv file is saved at [{}]'.format(csv_fn))
