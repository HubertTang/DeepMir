import numpy as np
import pandas as pd
from Bio import SeqIO
import scipy.stats as stats
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc
import random
import seaborn as sns

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import utils
import rna_dataset

random.seed(66)

def plot_normal(data, data_label, dir_name, fig_name, line_color='coral'):
    # plot and save the normal distribution of the given data
    m = np.mean(data)
    sig = np.std(data)

    x = np.linspace(-0.5, 1.5, 200)
    y = stats.norm.pdf(x, m, sig)

    # plt.plot(x, y, color=line_color)
    plt.plot(x, y, alpha=0.7, label=data_label)
    # plt.hist(data, bins=100, density=True, stacked=True, alpha=0.7)
    # plt.hist(data, bins=50, alpha=0.7)
    plt.grid()

    plt.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Normal Distribution')
    plt.legend()
    plt.savefig(f"{dir_name}/{fig_name}.png")
    plt.close('all')


def plot_dense(data, data_label, dir_name, fig_name):
    # plot and save the probability density of the given data
    plt.figure()
    sns.kdeplot(data, shade=True, label=data_label, alpha=.7)
    plt.grid()

    plt.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{dir_name}/{fig_name}.png")
    plt.close('all')


def to_percent(y, position):
    return str(100 * y) + '%'


def plot_hist(data, data_label, dir_name, fig_name):
    # plot and save the probability histgram of the given data
    # plt.figure()
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    # ax.hist(data, bins=100, density=True, alpha=0.7, label=data_label)
    # ax2.hist(data, bins=100, density=True, alpha=0.7)

    # formatter = FuncFormatter(to_percent)

    ax.hist(data, bins=100, weights= [1./ len(data)] * len(data), alpha=0.7, label=data_label)
    ax2.hist(data, bins=100, weights= [1./ len(data)] * len(data), alpha=0.7)

    # ax.gca().yaxis.set_major_formatter(formatter)
    # ax2.gca().yaxis.set_major_formatter(formatter)

    ax.set_ylim(0.8, 1.0)
    ax2.set_ylim(0, 0.1)


    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal    
    # sns.distplot(data, hist=True, bins=100, label=data_label)
    ax.grid()
    ax2.grid()

    # ax.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Percentage')
    ax.legend()
    plt.savefig(f"{dir_name}/{fig_name}.png")
    plt.close('all')



def softmax2npy(model_name, test_file, num_classes):
    model = load_model(model_name)
    model.summary()

    # load test dataset
    test_file_path = f"data/{test_file}"
    num_tests = utils.count_lines(test_file_path)
    test_generator = rna_dataset.RNA_onehot(test_file_path, num_tests, batch_size=256, dim=(200, 4),
                                            num_channels=1, num_classes=num_classes, shuffle=False)

    # prediction
    print('Start predicting ... ...')
    prediction = model.predict_generator(test_generator, workers=6,
                                         use_multiprocessing=True, verbose=1)
    # prediction = model.predict_generator(test_generator, verbose=1)
    np.save(f"{model_name.split('.').pop(0)}/{test_file.split('.').pop(0)}.npy", prediction)

    # release the memory occupied by GPU
    K.clear_session()


def out_plot(dir_name, array_file):
    arr = np.load(f"{dir_name}/{array_file}.npy")
    arr_1d = [np.max(r) for r in arr]
    # plot_normal(arr_1d, array_file, dir_name, 'test')
    print(np.max(arr_1d))
    # plot_dense(arr_1d, array_file, dir_name, 'test_dense')
    plot_hist(arr_1d, array_file, dir_name, 'test_broken_hist')


def draw_roc(model_name, posi_data, neg_data):
    # initialization
    num_seq = 1700
    # generate the label and score
    posi_arr = np.load(f"{model_name}/{posi_data}.npy")
    random.shuffle(posi_arr)
    posi_1d = [np.max(r) for r in posi_arr][:num_seq]
    neg_arr = np.load(f"{model_name}/{neg_data}.npy")
    random.shuffle(neg_arr)
    neg_1d = [np.max(r) for r in neg_arr][:num_seq]
    print(len(posi_1d), posi_1d[:10], len(neg_1d), neg_1d[:10])
    y_label = np.append(np.ones((num_seq,)), np.zeros((num_seq,)))
    posi_1d.extend(neg_1d)
    y_score = np.array(posi_1d)
    # print(y_label.shape, y_label[149998:150002], y_score.shape, y_score[149998:150002])

    # calculate the result of roc
    fpr, tpr, threshold = roc_curve(y_label, y_score)
    roc_auc = auc(fpr, tpr)
    
    # find the optimal cut off based on the Youden index
    # Youden Index = sensitiveity + specificity - 1
    # tpr = sensitivity
    # fpr = 1 - specificity
    youden_index = tpr - fpr
    optimal_thres = threshold[youden_index.argmax()]

    # plot the roc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label=f'ROC curve (area = {roc_auc:.5f})\n(cutoff = {optimal_thres:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}/ROC.png")
    plt.close('all')


def paper_plot(dir_name):
    # plot the miRNA with nagetive inputs
    # draw the precentage histgram
    mirna_arr = np.load(f"{dir_name}/mi_balance_60_50_test.npy")
    mirna_arr_1d = [np.max(r) for r in mirna_arr]

    other_arr = np.load(f"{dir_name}/neg_other.npy")
    other_arr_1d = [np.max(r) for r in other_arr]

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    # ax.hist(data, bins=100, density=True, alpha=0.7, label=data_label)
    # ax2.hist(data, bins=100, density=True, alpha=0.7)

    # formatter = FuncFormatter(to_percent)

    ax.hist(mirna_arr_1d, color='C1', bins=100, weights= [1./ len(mirna_arr_1d)] * len(mirna_arr_1d), alpha=0.6, label='miRNA')
    ax2.hist(mirna_arr_1d, color='C1', bins=100, weights= [1./ len(mirna_arr_1d)] * len(mirna_arr_1d), alpha=0.6)
    ax.hist(other_arr_1d, color='C2', bins=100, weights= [1./ len(other_arr_1d)] * len(other_arr_1d), alpha=0.6, label='negative')
    ax2.hist(other_arr_1d, color='C2', bins=100, weights= [1./ len(other_arr_1d)] * len(other_arr_1d), alpha=0.6)

    # ax.gca().yaxis.set_major_formatter(formatter)
    # ax2.gca().yaxis.set_major_formatter(formatter)

    # ax.set_ylim(0.8, 1.0)
    # ax2.set_ylim(0, 0.045)
    ax.set_ylim(0.6, 0.9)
    ax2.set_ylim(0, 0.06)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal    
    # sns.distplot(data, hist=True, bins=100, label=data_label)
    ax.grid()
    ax2.grid()

    # ax.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Percentage')
    ax.legend(loc="upper left")
    plt.savefig(f"{dir_name}/test_paper_hist.png")
    plt.close('all')

    # draw the roc curve
    num_seq = 1700
    # generate the label and score
    posi_arr = np.load(f"{dir_name}/mi_balance_60_50_test.npy")
    random.shuffle(posi_arr)
    # posi_1d = [np.max(r) for r in posi_arr][:num_seq]
    posi_1d = [np.max(r) for r in posi_arr]
    neg_arr = np.load(f"{dir_name}/neg_other.npy")
    random.shuffle(neg_arr)
    # neg_1d = [np.max(r) for r in neg_arr][:num_seq]
    neg_1d = [np.max(r) for r in neg_arr]
    print(len(posi_1d), posi_1d[:10], len(neg_1d), neg_1d[:10])
    # y_label = np.append(np.ones((num_seq,)), np.zeros((num_seq,)))
    y_label = np.append(np.ones((len(posi_1d),)), np.zeros((len(neg_1d),)))
    posi_1d.extend(neg_1d)
    y_score = np.array(posi_1d)
    # print(y_label.shape, y_label[149998:150002], y_score.shape, y_score[149998:150002])

    # calculate the result of roc
    fpr, tpr, threshold = roc_curve(y_label, y_score)
    roc_auc = auc(fpr, tpr)
    
    # find the optimal cut off based on the Youden index
    # Youden Index = sensitiveity + specificity - 1
    # tpr = sensitivity
    # fpr = 1 - specificity
    youden_index = tpr - fpr
    optimal_thres = threshold[youden_index.argmax()]
    optimal_index = 20500
    print(fpr[optimal_index], threshold[optimal_index])
    print("highest:", optimal_thres)

    # plot the roc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label=f'ROC curve (area = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random ROC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(model_name)
    plt.legend(loc="lower right")
    plt.savefig(f"{dir_name}/test_paper_ROC_2.png")
    plt.close('all')


def paper_plot_hist_line(dir_name):
    # plot the miRNA with nagetive inputs
    # draw the precentage histgram
    mirna_arr = np.load(f"{dir_name}/mi_balance_60_50_test.npy")
    mirna_arr_1d = [np.max(r) for r in mirna_arr]

    other_arr = np.load(f"{dir_name}/neg_other.npy")
    other_arr_1d = [np.max(r) for r in other_arr]

    bins = [0.01 * i for i in range(101)]

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    m, edges = np.histogram(mirna_arr_1d, bins)
    m = 100 * m / len(mirna_arr_1d)
    centers = 0.5 * (edges[1:] + edges[:-1])

    o, _ = np.histogram(other_arr_1d, bins)
    o = 100 * o / len(other_arr_1d)

    ax.plot(centers, m, color='C1', label='miRNA')
    ax2.plot(centers, m, color='C1')
    ax.plot(centers, o, color='blue', label='negative', linestyle="--")
    ax2.plot(centers, o, color='blue', linestyle="--")

    ax.set_ylim(60, 90)
    ax2.set_ylim(0, 6)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal    
    # sns.distplot(data, hist=True, bins=100, label=data_label)
    ax.grid()
    ax2.grid()

    # ax.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Percentage')
    ax.legend(loc="upper left")
    plt.savefig(f"{dir_name}/test_paper_hist_line.png")
    plt.close('all')


    
def paper_plot_others(dir_name):
    # plot the miRNA with nagetive inputs    
    mirna = np.load(f"{dir_name}/neg_mirna_60.npy")
    mirna_1d = [np.max(r) for r in mirna]

    trna = np.load(f"{dir_name}/neg_trna.npy")
    trna_1d = [np.max(r) for r in trna]

    cdbox = np.load(f"{dir_name}/neg_cdbox.npy")
    cdbox_1d = [np.max(r) for r in cdbox]
    
    plt.hist(mirna_1d, color='C1', bins=100, weights= [1./ len(mirna_1d)] * len(mirna_1d), alpha=0.6, label='untrained miRNA')
    plt.hist(trna_1d, color='C2', bins=100, weights= [1./ len(trna_1d)] * len(trna_1d), alpha=0.6, label='tRNA')
    plt.hist(cdbox_1d, color='C3', bins=100, weights= [1./ len(cdbox_1d)] * len(cdbox_1d), alpha=0.6, label='CDBOX')
    
    plt.grid()

    # ax.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Percentage')
    # ax.legend(loc="upper left")
    plt.legend()
    plt.savefig(f"{dir_name}/test_paper_hist_others.png")
    plt.close('all')


def paper_plot_line_others(dir_name):
    # plot the miRNA with nagetive inputs    
    mirna = np.load(f"{dir_name}/neg_mirna_60.npy")
    mirna_1d = [np.max(r) for r in mirna]

    trna = np.load(f"{dir_name}/neg_trna.npy")
    trna_1d = [np.max(r) for r in trna]

    cdbox = np.load(f"{dir_name}/neg_cdbox.npy")
    cdbox_1d = [np.max(r) for r in cdbox]

    bins = [0.01 * i for i in range(101)]
    
    m, edges = np.histogram(mirna_1d, bins)
    m = 100 * m / len(mirna_1d)
    centers = 0.5 * (edges[1:] + edges[:-1])

    t, _ = np.histogram(trna_1d, bins)
    t = 100 * t / len(trna_1d)

    c, _ = np.histogram(cdbox_1d, bins)
    c = 100 * c / len(cdbox_1d)

    plt.plot(centers, m, color='C1', label='Untrained miRNA')
    plt.plot(centers, t, color='C2', label='tRNA', linestyle="--")
    plt.plot(centers, c, color='C3', label='CDBOX', linestyle=":")
    
    plt.grid()

    # ax.title(fig_name)
    plt.xlabel('Probability')
    plt.ylabel('Percentage')
    # ax.legend(loc="upper left")
    plt.legend()
    plt.savefig(f"{dir_name}/test_paper_line_others.png")
    plt.close('all')


if __name__ == '__main__':
    model_n = 'mi_balance_60_50'
    # test_file_ba_list = ['mi_balance_60_50_test', 'neg_mirna_60', 'neg_trna', 'neg_cdbox', 'neg_other']   
    # test_file_ba_list = ['mi_imbalance_60_50_test', 'neg_mirna_60', 'neg_trna', 'neg_cdbox', 'neg_other']   
    test_file_name = 'mi_balance_60_50_test'
    test_file_ba_list = ['mi_balance_60_50_test', 'neg_other']

    try:
        os.mkdir(model_n)
    except FileExistsError:
        pass

    # # generate prediction result from a list of dataset
    # for test_file_name in test_file_ba_list:
    #     softmax2npy(model_name=f'{model_n}.h5',
    #                  test_file=f'{test_file_name}.csv',
    #                  num_classes=165)

    # # generate prediction result from one specific dataset file
    # softmax2npy(model_name=f'{model_n}.h5',
    #              test_file=f'{test_file_name}.csv',
    #              num_classes=165)

    # # out_plot(test_file_name)
    # for plot_arr in test_file_ba_list:
    #     out_plot(model_n, plot_arr)

    # draw_roc(model_n, 'mi_balance_60_50_test', 'neg_other')

    # temp test
    # out_plot(model_n, test_file_name)

    paper_plot(model_n)
    # paper_plot_others(model_n)
    # paper_plot_line_others(model_n)
    # paper_plot_hist_line(model_n)
