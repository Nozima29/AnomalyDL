import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import conf
import model
import numpy as np


def swat_time_to_nanosec(t: str) -> int:
    return int(datetime.strptime(t, conf.SWaT_TIME_FORMAT).timestamp() * 1_000_000_000)


class Network:
    def __init__(self, n_features: int, n_hiddens: int):
        self.n_features = n_features
        # self.pidx = pidx
        # self.gidx = gidx
        #self.gpu = torch.device('cuda:{}'.format(gidx - 1))
        self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = model.Encoder(
            n_inputs=n_features, n_hiddens=n_hiddens).to(self.gpu)
        self.decoder = model.AttentionDecoder(
            n_hiddens=n_hiddens, n_features=n_features).to(self.gpu)

        self.model_fn = 'checkpoints/SWaT-P{}'.format(1)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), amsgrad=True)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), amsgrad=True)

        self.mse_loss = nn.MSELoss()

    def train(self, dataset, batch_size) -> float:
        epoch_loss = 0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            answer, guess = self.infer(batch)
            loss = self.mse_loss(guess, answer)
            loss.backward()
            epoch_loss += loss.item()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        return epoch_loss

    def eval(self, dataset, batch_size) -> float:
        epoch_loss = 0
        n_datapoints = 0
        tms, distances, labels = [], [], []
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            ts = [swat_time_to_nanosec(t) for t in batch['ts']]
            attack = batch['attack']
            labels.append(attack)
            answer, guess = self.infer(batch)
            # distance = torch.norm(
            #     guess - answer, p=conf.EVALUATION_NORM_P, dim=1)
            # distances.append(distance)
            distances.append(torch.abs(answer - guess).cpu().numpy())
            #epoch_loss += torch.sum(distance).item()
            n_datapoints += len(ts)
            tms.append(batch['ts'])
        context = {
            # 'loss': epoch_loss / n_datapoints,
            'dist': np.concatenate(distances),
            'ts': np.concatenate(tms),
            'label': np.concatenate(labels)
        }
        return context

    def infer(self, batch):
        # print(self.gpu)
        given = batch['given'].to(self.gpu)
        predict = batch['predict'].to(self.gpu)
        answer = batch['answer'].to(self.gpu)
        encoder_outs, context = self.encoder(given)
        guess = self.decoder(encoder_outs, context, predict)
        return answer, guess

    def load(self, idx: int) -> float:
        fn = self.model_fn + '-{}.net'.format(idx)
        checkpoint = torch.load(fn)
        self.encoder.load_state_dict(checkpoint['model_encoder'])
        self.decoder.load_state_dict(checkpoint['model_decoder'])
        self.encoder_optimizer.load_state_dict(checkpoint['optimizer_encoder'])
        self.decoder_optimizer.load_state_dict(checkpoint['optimizer_decoder'])
        return checkpoint['min_loss']

    def save(self, idx: int, min_loss: float) -> None:
        fn = self.model_fn + '-{}.net'.format(idx)
        torch.save(
            {
                'min_loss': min_loss,
                'model_encoder': self.encoder.state_dict(),
                'model_decoder': self.decoder.state_dict(),
                'optimizer_encoder': self.encoder_optimizer.state_dict(),
                'optimizer_decoder': self.decoder_optimizer.state_dict(),
            },
            fn
        )

    def train_mode(self) -> None:
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self) -> None:
        self.encoder.eval()
        self.decoder.eval()
