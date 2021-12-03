import torch
import torch.nn.parallel
from torch import nn
from datetime import datetime
import conf
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader


def swat_time_to_nanosec(t: str) -> int:
    return int(datetime.strptime(t, conf.SWaT_TIME_FORMAT).timestamp() * 1_000_000_000)


class Encoder(nn.Module):
    def __init__(self, n_inputs, n_hiddens):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.lstm1 = nn.LSTMCell(input_size=n_inputs, hidden_size=n_hiddens)
        self.lstm2 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.lstm3 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.lstm4 = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)

    def init_hidden_and_cell_state(self, batch_size, dev):
        return (
            torch.zeros((batch_size,
                         self.n_hiddens),
                        dtype=torch.float,
                        device=dev),
            torch.zeros((batch_size,
                         self.n_hiddens),
                        dtype=torch.float,
                        device=dev)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_gpu = 'cpu'  # torch.device('cuda:{}'.format(x.get_device()))
        hc1 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc2 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc3 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        hc4 = self.init_hidden_and_cell_state(batch_size, x_gpu)
        x = x.view(batch_size, conf.WINDOW_GIVEN, -1)
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        encoder_outs = []
        for point in x:
            hc1 = self.lstm1(point, hc1)
            hc2 = self.lstm2(hc1[0], hc2)
            hc3 = self.lstm3(hc2[0], hc3)
            hc4 = self.lstm4(hc3[0], hc4)
            encoder_outs.append(hc4[0].unsqueeze(1))
        out = torch.cat(encoder_outs, dim=1)  # (B, seq, hiddens)
        return out, hc4[1]


class AttentionDecoder(nn.Module):
    def __init__(self, n_hiddens, n_features):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.lstm = nn.LSTMCell(input_size=n_hiddens, hidden_size=n_hiddens)
        self.extend = nn.Linear(n_features, n_hiddens)
        self.attn = nn.Sequential(
            nn.Linear(n_hiddens * 2, conf.WINDOW_GIVEN), nn.Softmax(dim=1))
        self.attn_combine = nn.Sequential(
            nn.Linear(n_hiddens * 2, n_hiddens), nn.ReLU())
        self.out = nn.Linear(n_hiddens, n_features)

    def init_cell_state(self, batch_size, dev):
        return torch.zeros((batch_size, self.n_hiddens), dtype=torch.float, device=dev)

    def forward(self, encoder_outs, context, predict):
        batch_size = predict.size(0)
        x_gpu = 'cpu'  # torch.device('cuda:{}'.format(predict.get_device()))
        # (batch, seq, params) -> (seq, batch, params)
        predict = predict.transpose(0, 1)
        h_i = context
        c_i = self.init_cell_state(batch_size, x_gpu)
        for idx in range(conf.WINDOW_PREDICT):
            inp = self.extend(predict[idx])
            # input:(B, params -> hiddens), c_i:(B, hiddens) => attn_weights:(B, window-size given)
            attn_weights = self.attn(torch.cat((inp, context), dim=1))
            # BMM of attn_weights (B, 1, windows given) and encoding outputs (B, seq, hiddens) -> (B, 1, hiddens)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outs)
            attn_combine = self.attn_combine(
                torch.cat((inp, attn_applied.squeeze(1)), dim=1))
            h_i, c_i = self.lstm(attn_combine, (h_i, c_i))
            # h_i : (B, hiddens), self.out : (B, hiddens) -> (B, # of features)
        return self.out(h_i)  # (B, n_outs, # of features)


class Network:
    def __init__(self, n_features: int, n_hiddens: int):
        self.n_features = n_features
        # self.pidx = pidx
        # self.gidx = gidx
        #self.gpu = torch.device('cuda:{}'.format(gidx - 1))
        self.gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = Encoder(
            n_inputs=n_features, n_hiddens=n_hiddens).to(self.gpu)
        self.decoder = AttentionDecoder(
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
            distance = torch.norm(
                guess - answer, p=conf.EVALUATION_NORM_P, dim=1)
            #distances.append(torch.abs(answer - guess).cpu().numpy())
            distances.append(distance)
            epoch_loss += torch.sum(distance).item()
            n_datapoints += len(ts)
            tms.append(batch['ts'])
        context = {
            'loss': epoch_loss/n_datapoints,
            'dist': np.concatenate(distances),
            'ts': np.concatenate(tms),
            'label': np.concatenate(labels)
        }
        return context

    def infer(self, batch):
        given = batch['given'].to(self.gpu)
        predict = batch['predict'].to(self.gpu)
        answer = batch['answer'].to(self.gpu)
        encoder_outs, context = self.encoder(given)
        guess = self.decoder(encoder_outs, context, predict)
        print(guess.shape)
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
