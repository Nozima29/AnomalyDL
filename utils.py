from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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


def check_graph(xs, att, piece=2, THRESHOLD=None):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD != None:
            axs[i].axhline(y=THRESHOLD, color='r')
    plt.show()


def get_score(score, groud_truth):
    c_score = range_check(score, size=30)

    auc = roc_auc_score(groud_truth, score)
    print('ROC-AUC score:',  auc)

    plt.figure(figsize=(15, 5))
    plt.plot(score[:500], label="before")
    plt.plot(c_score[:500], label="after")
    plt.legend()
    plt.show()


def plot_roc():
    seq_score = pd.read_csv('seq.csv')
    enc_score = pd.read_csv('enc.csv')
    trans_score = pd.read_csv('trans.csv')

    fpr1, tpr1, th1 = roc_curve(seq_score['label'], seq_score['score'])
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, th2 = roc_curve(enc_score['label'], enc_score['score'])
    roc_auc2 = auc(fpr2, tpr2)

    fpr3, tpr3, th3 = roc_curve(trans_score['label'], trans_score['score'])
    roc_auc3 = auc(fpr3, tpr3)

    optimal_idx = np.argmax(tpr3 - fpr3)
    optimal_threshold = th3[optimal_idx]
    print("Threshold value is:", optimal_threshold)

    sd = trans_score.loc[trans_score['score'] > optimal_threshold]
    correct = sd.loc[sd['label'] == 1]
    print(len(sd), len(correct))

    # plt.figure()
    # plt.plot(
    #     fpr1,
    #     tpr1,
    #     color="darkorange",
    #     label="ROC curve for Seq2seq (area = %0.2f)" % roc_auc1,
    # )

    # plt.plot(
    #     fpr2,
    #     tpr2,
    #     color="darkblue",
    #     label="ROC curve for LSTMEncoder (area = %0.2f)" % roc_auc2,
    # )

    # plt.plot(
    #     fpr3,
    #     tpr3,
    #     color="darkred",
    #     label="ROC curve for Transformer (area = %0.2f)" % roc_auc3,
    # )

    # plt.xlim([-0.1, 1.0])
    # plt.ylim([-0.1, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic")
    # plt.legend(loc="lower right")
    # plt.show()


# plot_roc()
