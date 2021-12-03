from seq2seq_model import Network
from swat_dataset import SWaTDataset
from datetime import datetime
import conf
from conf import args
import torch
from transformer_model import Transformer
from torch.utils.data import DataLoader
import numpy as np
from utils import get_score


def eval_model(type):
    #p_features = len(conf.P_SRCS[4 - 1])
    loss = 0
    eval_dataset = SWaTDataset(
        'data/dat/swat-test-P{}.dat'.format(1), attack=True)

    if type == 'trans':
        model = Transformer(args).to(args.device)

        with open('transfomer', "rb") as f:
            saved_state_dict = torch.load(f)
        model.load_state_dict(saved_state_dict)

        dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
        ts, dist = [], []
        losses, labels = 0, []
        n_datapoints = 0

        start = datetime.now()
        with torch.no_grad():
            for batch in dataloader:
                given = batch["given"]
                answer = batch["answer"]
                labels.append(batch['attack'])
                ts.append(np.array(batch["ts"]))
                guess = model(given.to(args.device), given.to(args.device))
                distance = torch.norm(
                    guess - answer.to(args.device), p=conf.EVALUATION_NORM_P, dim=1)
                losses += torch.sum(distance).item()
                dist.append(
                    torch.abs(answer.to(args.device) - guess).cpu().numpy())
                n_datapoints += len(ts)
        loss = losses/n_datapoints
        print(f'val loss: {loss} ({datetime.now() - start})')
        context = {
            'loss': loss,
            'dist': np.concatenate(dist),
            'ts': np.concatenate(ts),
            'label': np.concatenate(labels)
        }

    else:
        model = Network(n_features=2, n_hiddens=conf.N_HIDDEN_CELLS)
        model.load(1)
        model.eval_mode()
        start = datetime.now()
        with torch.no_grad():
            context = model.eval(eval_dataset, conf.BATCH_SIZE)
            loss = context['loss']
        print(f'val loss: {loss} ({datetime.now() - start})')

    score, label = context['dist'], context['label']
    get_score(score, label)


if __name__ == '__main__':
    model_type = 'seq'  # 'trans'
    eval_model(model_type)
