from autoencoder import RecurrentAutoencoder
from torch.utils.data import DataLoader
from swat_dataset import SWaTDataset
from transformer import Transformer
from datetime import datetime
from seq2seq import Network
from utils import get_score
from conf import args
import numpy as np
import torch
import conf


def eval_model(type):
    eval_dataset = SWaTDataset(
        'data/dat/swat-test-P{}.dat'.format(1), attack=True)
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    dist = []
    labels = []

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
                guess = model(given.to(args.device), given.to(args.device))
                dist.append(
                    torch.abs(answer.to(args.device) - guess).cpu().numpy())
        d = np.concatenate(dist)
        score = np.mean(d, axis=1)
        context = {
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
        preds, trues = [], []
        with torch.no_grad():
            model.eval()
            for batch in dataloader:
                input_data = batch['given']
                target = batch['answer']
                labels.append(batch['attack'])
                pred = model(input_data.to(conf.args.device))
                preds.append(pred.cpu().numpy())
                trues.append(target.cpu().numpy())
                dist.append(
                    torch.abs(target.to(conf.args.device) - pred).cpu().numpy())
        preds = np.max(np.concatenate(preds), axis=1)
        trues = np.max(np.concatenate(trues), axis=1)
        d = np.concatenate(dist)
        score = np.mean(d, axis=1)
        context = {
            'dist': score,
            'label': np.concatenate(labels)
        }

    score, label = context['dist'], context['label']
    get_score(score, label)


if __name__ == '__main__':
    """
    Evaluate any of 3 models
        'trans'=>transformer
        'seq'=>seq2seq
        'enc'=>autoencoder
    """
    model_type = 'seq'
    eval_model(model_type)
