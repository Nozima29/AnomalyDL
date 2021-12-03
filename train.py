from seq2seq_model import Network
from swat_dataset import SWaTDataset
from datetime import datetime
import conf
from conf import args
import sys
import torch
from transformer_model import Transformer
from autoencoder import RecurrentAutoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy


def train_model(type):
    train_dataset = SWaTDataset(
        'data/dat/swat-train-P{}.dat'.format(1), attack=False)

    if type == 'trans':
        model = Transformer(args).to(args.device)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in tqdm(range(1, args.epochs+1)):
            total_loss = 0
            n = 0
            for data in train_dataloader:
                input_data = data['given']
                target = data['answer']
                model.train()
                optimizer.zero_grad()
                pred = model(input_data.to(args.device),
                             input_data.to(args.device))
                loss = loss_function(pred, target.to(args.device))
                loss.backward()
                optimizer.step()
                total_loss += (loss*input_data.size()[0])
                n += input_data.size()[0]

            print('epoch: {}, Train MSE_loss : {:.5f}'.format(epoch, total_loss/n))

        torch.save(model.state_dict(), './checkpoints/transfomer')

    elif type == 'seq':
        model = Network(n_features=2, n_hiddens=conf.N_HIDDEN_CELLS)
        min_loss = sys.float_info.max
        for e in range(args.epochs):
            start = datetime.now()
            loss = model.train(train_dataset, conf.BATCH_SIZE)

            if loss < min_loss:
                min_loss = loss
                model.save(1, min_loss)
            print(f'[{e+1:>4}] {loss:10.6} ({datetime.now() - start})')

    else:
        model = RecurrentAutoencoder(
            train_dataset[0]['given'].shape[0], train_dataset[0]['given'].shape[1])
        model = model.to(conf.args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        history = dict(train=[], val=[])
        train_dataloader = DataLoader(
            train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True)

        for epoch in range(1, conf.args.epochs + 1):
            model = model.train()
            train_losses = []
            for data in train_dataloader:
                input_data = data['given']
                target = data['answer']
                optimizer.zero_grad()
                pred = model(input_data.to(conf.args.device))
                loss = criterion(pred, target.to(conf.args.device))

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_loss = np.mean(train_losses)

                history['train'].append(train_loss)

            print(f'Epoch {epoch}: train loss {train_loss}')

        torch.save(model.state_dict(), './checkpoints/encoder')


if __name__ == '__main__':
    model_type = 'enc'  # 'trans', 'seq'
    train_model(model_type)
