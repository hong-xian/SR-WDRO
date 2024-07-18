import torch
import torch.nn.functional as F
import torch.nn as nn


def attack_udr(model, x, y, args, lamda, early_stop=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(x).cuda()
    kl = nn.KLDivLoss(reduction='batchmean')
    for _ in range(args.random_restarts):
        delta = torch.zeros_like(x).cuda()
        if args.constraint == "Linf":
            delta.uniform_(-args.eps, args.eps)
        elif args.constraint == "L2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*args.eps
        else:
            raise ValueError

        delta = torch.clamp(delta, 0-x, 1-x)
        delta.requires_grad = True
        for _ in range(args.num_steps):
            output = model(x + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if args.udr_type == "AT":
                loss = F.cross_entropy(output, y)
            elif args.udr_type == "Trades":
                logits_cln = model(x).detach().requires_grad_(False)
                loss = kl(F.log_softmax(output, dim=1), F.softmax(logits_cln, dim=1))
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = x[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + args.step_size * torch.sign(g)
            # equal x_adv = x_adv + args.step_size * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = args.step_size
            # Simply choose tau = args.step_size

            tau = args.tau_m * args.step_size

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            d = d - lamda.detach() * args.step_size / tau * (d - torch.sign(d) * args.eps) * (abs_d > args.eps)

            d = torch.clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(x+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
