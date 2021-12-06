from seq2seq import Network
from swat_dataset import SWaTDataset
from datetime import datetime
import conf
from conf import args
import torch
from transformer import Transformer
from autoencoder import RecurrentAutoencoder
from torch.utils.data import DataLoader
import numpy as np
from utils import get_score


def eval_model(type):
    eval_dataset = SWaTDataset(
        'data/dat/swat-test-P{}.dat'.format(1), attack=True)
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    ts, dist = [], []
    losses, labels = 0, []
    loss = 0
    n_datapoints = 0

    if type == 'trans':
        model = Transformer(args).to(args.device)

        with open('transfomer', "rb") as f:
            saved_state_dict = torch.load(f)
        model.load_state_dict(saved_state_dict)

        start = datetime.now()
        with torch.no_grad():
            model.eval()
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
        d = np.concatenate(dist)
        score = np.mean(d, axis=1)
        context = {
            'loss': loss,
            'dist': score,
            'label': np.concatenate(labels)
        }

    elif type == 'seq':
        model = Network(n_features=2, n_hiddens=conf.N_HIDDEN_CELLS)
        model.load(1)
        model.eval_mode()
        start = datetime.now()
        with torch.no_grad():
            context = model.eval(eval_dataset, conf.BATCH_SIZE)
            loss = context['loss']
        print(f'val loss: {loss} ({datetime.now() - start})')

    else:
        model = RecurrentAutoencoder(
            eval_dataset[0]['given'].shape[0], eval_dataset[0]['given'].shape[1])

        with open('./checkpoints/encoder', "rb") as f:
            saved_state_dict = torch.load(f)
        model.load_state_dict(saved_state_dict)
        start = datetime.now()
        with torch.no_grad():
            model.eval()
            for batch in dataloader:
                input_data = batch['given']
                target = batch['answer']
                labels.append(batch['attack'])
                ts.append(np.array(batch["ts"]))
                pred = model(input_data.to(conf.args.device))
                # distance = torch.norm(
                #     pred - target.to(conf.args.device), p=conf.EVALUATION_NORM_P, dim=1)
                # loss += torch.sum(distance).item()
                dist.append(
                    torch.abs(target.to(conf.args.device) - pred).cpu().numpy())
                n_datapoints += len(ts)
        #print(f'val loss: {loss} ({datetime.now() - start})')
        #loss = loss/n_datapoints
        d = np.concatenate(dist)
        score = np.mean(d, axis=1)
        context = {
            #   'loss': loss,
            'dist': score,
            'label': np.concatenate(labels)
        }

    score, label = context['dist'], context['label']
    import pandas as pd
    df = pd.DataFrame({'score': score, 'label': label})
    df.to_csv('{}.csv'.format(type), index=None)
    #get_score(score, label)


if __name__ == '__main__':
    model_type = 'seq'  # 'trans'
    eval_model(model_type)
