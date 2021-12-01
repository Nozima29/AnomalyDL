from swat_dataset import SWaTDataset
from network import Network
from datetime import datetime
import conf
import sys
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def range_check(series, size):
    size = size
    data = []

    for i in range(len(series)-size+1):
        if i == 0:
            check_std = np.std(series[i:i+size])
        std = np.std(series[i:i+size])
        mean = np.mean(series[i:i+size])
        max = np.max(series[i:i+size])
        if check_std * 2 >= std:
            check_std = std
            data.append(mean)
        elif max == series[i]:
            data.append(max*5)
            check_std = std
        else:
            data.append(series[i]*3)
    for _ in range(size-1):
        data.append(mean)

    return np.array(data)


def train_model():
    p_features = len(conf.P_SRCS[3 - 1])
    train_dataset = SWaTDataset(
        'data/dat/swat-train-P{}.dat'.format(1), attack=False)
    pnet = Network(n_features=2, n_hiddens=conf.N_HIDDEN_CELLS)
    epochs = 10
    min_loss = sys.float_info.max
    for e in range(epochs):
        start = datetime.now()
        loss = pnet.train(train_dataset, conf.BATCH_SIZE)

        if loss < min_loss:
            min_loss = loss
            pnet.save(1, min_loss)
        print(f'[{e+1:>4}] {loss:10.6} ({datetime.now() - start})')
        # print(f'* the total training time: {datetime.now() - training_start}')


def eval_model():
    p_features = len(conf.P_SRCS[4 - 1])
    eval_dataset = SWaTDataset(
        'data/dat/swat-test-P{}.dat'.format(1), attack=True)
    pnet = Network(n_features=2, n_hiddens=conf.N_HIDDEN_CELLS)
    pnet.load(1)
    pnet.eval_mode()
    start = datetime.now()
    with torch.no_grad():
        context = pnet.eval(eval_dataset, conf.BATCH_SIZE)
    # print(f'* val loss: {context['loss']} ({datetime.now() - start})')
    thrsh_up = 0.006865
    thrsh_low = 0.00686282  # 0.00686  # 0.002384
    df = pd.DataFrame(
        {'ts': context['ts'], 'dist': np.mean(context['dist']), 'label': context['label']})
    sd = df.loc[(df['dist'] < thrsh_up) & (df['dist'] > thrsh_low)]
    correct = len(sd.loc[sd['label'] == 1])
    all = len(sd)
    # print(correct/all)

    print(df.describe())
    # print(df.describe())
    # sns.distplot(context['dist'], bins=50, kde=True)
    # plt.show()


if __name__ == '__main__':
    # train_model()
    eval_model()
