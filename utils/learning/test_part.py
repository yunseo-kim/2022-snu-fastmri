import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

import sys


def log_path(args):
    log_f = open(args.exp_dir / 'val-log.txt', 'w')
    return log_f


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    log_f = log_path(args)

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())
    print('Current cuda device ', torch.cuda.current_device(), file=log_f)

    model = VarNet(num_cascades=args.cascade, pools=4, chans=18, sens_pools=4, sens_chans=8)
    model.to(device=device)

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print("epoch: ", checkpoint['epoch'], "best_val_loss: ", checkpoint['best_val_loss'].item())
    print("epoch: ", checkpoint['epoch'], "best_val_loss: ", checkpoint['best_val_loss'].item(), file=log_f)
    model.load_state_dict(checkpoint['model'])

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

    log_f.close()