from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List
import pandas as pd
import numpy as np
import conf
import csv

DATA = 'data/swat-test.csv'


class Normalizer:
    def __init__(self):
        self.LR = {}
        df = pd.read_csv(DATA)
        df = df.rename(columns=lambda x: x.strip())
        for col in conf.ALL_SRCS:
            desc = df[col].describe()
            self.LR[col] = (desc['min'], desc['max'])

    def normalize(self, dic: Dict, col: str) -> float:
        cv = float(dic[col])
        if col in conf.DIGITAL_SRCS:
            return cv / 2  # 0, 0.5, 1
        L, R = self.LR[col]
        return (cv - L) / (R - L)


def attack_window_p(w):
    for att_p in w:
        if att_p:
            return True
    return False


def data_generator(filename: str):
    normalizer = Normalizer()
    with open(filename) as fh:
        reader = csv.reader(fh, delimiter=',', quoting=csv.QUOTE_NONE)
        header = list(map(lambda x: x.strip(), next(reader)))
        q_ts: deque = deque(maxlen=conf.WINDOW_SIZE)
        q_signals: List[deque] = [
            deque(maxlen=conf.WINDOW_SIZE) for _ in range(conf.N_PROCESS)]
        q_attack: deque = deque(maxlen=conf.WINDOW_SIZE)
        for line in reader:
            line = list(map(lambda x: x.strip(), line))
            # make a map from header's field names
            dic = dict(zip(header, line))
            q_ts.append(datetime.strptime(
                dic[conf.HEADER_STRING_TIMESTAMP], conf.SWaT_TIME_FORMAT))
            for pidx in range(conf.N_PROCESS):
                q_signals[pidx].append(
                    np.array([normalizer.normalize(dic,
                                                   x) for x in conf.P_SRCS[pidx]],
                             dtype=np.float32)
                )
            q_attack.append(True if dic[conf.HEADER_STRING_NORMAL_OR_ATTACK].strip(
            ).upper() == 'ATTACK' else False)
            # print(q_attack)
            if len(q_ts) == conf.WINDOW_SIZE:
                if q_ts[0] + timedelta(seconds=conf.WINDOW_SIZE - 1) != q_ts[-1]:
                    continue
                signals_window = [np.array(q_signals[pidx])
                                  for pidx in range(conf.N_PROCESS)]
                split_window = [
                    (w[:conf.WINDOW_GIVEN],
                     #w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                     w[-1]) for w in signals_window
                ]
                yield q_ts[conf.WINDOW_GIVEN], split_window, attack_window_p(q_attack)


def main():
    g = data_generator(DATA)
    lines = 0
    lines_attack = 0

    picks = [[] for _ in range(conf.N_PROCESS)]
    while True:
        try:
            ts, window, is_attack = next(g)
            for pidx in range(conf.N_PROCESS):
                given, prediction, answer = window[pidx]
                picks[pidx].append(
                    [ts, given, prediction, answer, is_attack])
                print(is_attack)
            lines += 1
            if is_attack:
                lines_attack += 1

        except StopIteration:
            break
    # for pidx in range(conf.N_PROCESS):
    # out_filename = 'data/dat/{}-P{}.dat'.format('swat-test', 1)
    # with open(out_filename, "wb") as fh:
    #     pickle.dump(picks[pidx], fh)
    # print(f'* writing to {out_filename}')
    # print(
    #     f'* {lines:,} data-points have been written ({lines_attack:,} attack windows)')


def prepare(ex_file):
    from datetime import datetime
    df1 = pd.read_csv(ex_file)
    df = df1.copy()
    rows = len(df.values)-1
    i = 0
    norm_date = []
    for col in df.values:
        str_date = datetime.strptime(col[0], '%d/%m/%Y %I:%M:%S.%f %p')
        str_date = str_date.replace(microsecond=0)
        str_date = str_date.strftime('%d/%m/%Y %I:%M:%S %p')
        norm_date.append(str_date)

    df = df.drop(df.columns[[0]], axis=1)
    df.insert(loc=0, column='t_stamp', value=norm_date)
    df.to_csv('swat-normal.csv', index=None)


if __name__ == '__main__':
    main()
