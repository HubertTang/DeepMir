import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def cmd_onehot():
    parser = argparse.ArgumentParser(description="ARGUMENTS")

    # argument for dataset
    parser.add_argument(
        'dataset',
        type=str,
        help="Dataset directory (MUST in `inputdir` directory)"
    )

    parser.add_argument(
        "num_classes",
        type=int,
        help="Number of families")

    parser.add_argument(
        "--seq_length",
        default=200,
        type=int,
        help="Length of RNA sequence (default: 200)")

    # argument for training
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default='DeepRfam',
    #     help="Choose model (DeepRfam(default), DeepRfam_deep, DeepRfam_lenet, ImgFam, L4Fam, L4BNFam, L5CFam, L5Fam, "
    #          "L5CFam_nopooling, L5CFam_dilation, L5CFam_ave, L5CFam_temp, L6Fam, L7CFam, Github_scnn)"
    # )

    # parser.add_argument(
    #     '--encode',
    #     type=str,
    #     default='RNA_onehot',
    #     help="Choose encoding method (RNA_onehot(default), RNA_img, RNA_fimg, RNA_pimg)"
    # )

    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch size (default: 64)'
    )

    parser.add_argument(
        '--num_epochs',
        default=5,
        type=int,
        help="Number of epochs to train (default: 5)"
    )

    # parser.add_argument(
    #     '--train_file',
    #     type=str,
    #     default=f'train.csv',
    #     help="Input data for training (default: train.csv)"
    # )
    #
    # parser.add_argument(
    #     '--valid_file',
    #     type=str,
    #     default=f'validation.csv',
    #     help="Input data for validation (default: validation.csv)"
    # )
    #
    # parser.add_argument(
    #     '--test_file',
    #     type=str,
    #     default='test.csv',
    #     help="Input data for training (default: test.csv)"
    # )
    #
    # parser.add_argument(
    #     '--family_dict_file',
    #     type=str,
    #     default='fam_label.csv',
    #     help="Input data for training (default: fam_label.csv)"
    # )
    #
    parser.add_argument(
        '--filter_sizes',
        default=[2, 4, 6, 8, 10, 12, 14, 16],
        type=int,
        nargs='+',
        help="Space separated list of motif filter lengths. (ex, --filter_sizes 4 8 12)\
            \n(default: [2, 4, 6, 8, 10, 12, 14, 16])"
    )

    parser.add_argument(
        '--num_filters',
        default=256,
        type=int,
        help='Number of filters per kernel (default: 256)'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.7,
        help='Rate to be kept for dropout. (default: 0.7)'
    )

    parser.add_argument(
        '--num_hidden',
        type=int,
        default=512,
        help='Number of neurons in hidden layer. (default: 512)'
    )

    parser.add_argument(
        '--loss_function',
        type=str,
        default="categorical_crossentropy",
        help='Loss function. (default: categorical_crossentropy)'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default="Adam",
        help='Optimizer. (default: Adam)'
    )

    # for logging
    parser.add_argument(
        '--log_name',
        type=str,
        default=None,
        help="Name for logging. (default: local_time)"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="log",
        help="Directory for logging. (default: log)"
    )

    # parser.add_argument(
    #     '--remark',
    #     type=str,
    #     default=None,
    #     help="Remark additional information"
    # )

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='version_2.0'
    )

    args = parser.parse_args()

    # assert (args.model in ['DeepRfam', 'L5CFam'])
    # assert (args.encode.split('#').pop(0) in ['RNA_onehot', 'RNA_img'])

    return args


def cmd_img():
    parser = argparse.ArgumentParser(description="ARGUMENTS")

    # argument for dataset
    parser.add_argument(
        'dataset',
        type=str,
        help="Dataset directory (MUST in `inputdir` directory)"
    )

    parser.add_argument(
        "num_classes",
        type=int,
        help="Number of families")

    parser.add_argument(
        "--seq_length",
        default=200,
        type=int,
        help="Length of RNA sequence (default: 200)")

    # argument for training
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default='DeepRfam',
    #     help="Choose model (DeepRfam(default), DeepRfam_deep, DeepRfam_lenet, ImgFam, L4Fam, L4BNFam, L5CFam, L5Fam, "
    #          "L5CFam_nopooling, L5CFam_dilation, L5CFam_ave, L5CFam_temp, L6Fam, L7CFam, Github_scnn)"
    # )

    # parser.add_argument(
    #     '--encode',
    #     type=str,
    #     default='RNA_onehot',
    #     help="Choose encoding method (RNA_onehot(default), RNA_img, RNA_fimg, RNA_pimg)"
    # )

    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--num_epochs',
        default=5,
        type=int,
        help="Number of epochs to train (default: 5)"
    )

    parser.add_argument(
        '--filter_sizes',
        default=[2, 2],
        type=int,
        nargs='+',
        help="Space separated list of motif filter lengths. (e.g., --filter_sizes 3 5) (default: [2, 2])"
    )

    parser.add_argument(
        '--num_filters',
        default=[32, 64],
        type=int,
        nargs='+',
        help='Number of filters per kernel in two convolution layers. (e.g., --num_filters 16 32) (default: [32, 64])'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help='Rate to be kept for dropout. (default: 0.7)'
    )

    parser.add_argument(
        '--num_hidden',
        type=int,
        default=[128, 64],
        help='Number of neurons in first two hidden layers. (e.g., --num_hidden 64 32) (default: [128, 64])'
    )

    parser.add_argument(
        '--loss_function',
        type=str,
        default="categorical_crossentropy",
        help='Loss function. (default: categorical_crossentropy)'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default="Adam",
        help='Optimizer. (default: Adam)'
    )

    # for logging
    parser.add_argument(
        '--log_name',
        type=str,
        default=None,
        help="Name for logging. (default: local_time)"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="log",
        help="Directory for logging. (default: log)"
    )

    # parser.add_argument(
    #     '--remark',
    #     type=str,
    #     default=None,
    #     help="Remark additional information"
    # )

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='version_2.0'
    )

    args = parser.parse_args()

    # assert (args.model in ['DeepRfam', 'L5CFam'])
    # assert (args.encode.split('#').pop(0) in ['RNA_onehot', 'RNA_img'])

    return args


def count_lines(file_name):
    with open(file_name) as f:
        line_count = sum(1 for line in f if line.strip())
    return line_count


def plot_history(history, save_dir):
    """https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search/data
    """

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_dir}/loss.png")
    plt.close('all')

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy.png")
    plt.close('all')


def cm2csv(true_labels, predicted_labels, dict_file, save_dir):
    """Generate confusion matrix and save it to .csv file
    `true_labels`, `predicted_labels` should be 1-D array
    """

    df = pd.read_csv(dict_file, header=None)
    fam_list = [df[0][i] for i in range(len(df[0]))]

    cm = metrics.confusion_matrix(true_labels, predicted_labels)
    frame = pd.DataFrame(cm, columns=fam_list, index=fam_list)
    frame.to_csv(f"{save_dir}/cm.csv", sep=',')

    # print(metrics.classification_report(true_labels, predicted_labels, target_names=fam_list))


def plot_cm(true_labels, predicted_labels, dict_file, title, save_dir, cm=None, normalize=None):
    """Visualize the confusion matrix.
    """

    df = pd.read_csv(dict_file, header=None)
    fam_list = [df[0][i] for i in range(len(df[0]))]

    if cm:
        cm = cm
    else:
        cm = metrics.confusion_matrix(true_labels, predicted_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    cmap = plt.cm.Blues
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_indexs = np.arange(len(fam_list))
    plt.xticks(tick_indexs, fam_list, rotation=45)
    plt.yticks(tick_indexs, fam_list)

    thres = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         ha='center', va='center',
                         color="white" if cm[i, j] > thres else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         ha='center', va='center',
                         color="white" if cm[i, j] > thres else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close('all')


def classification_report(true_labels, predicted_labels, dict_file, save_dir, std_out=True):
    """ Show the classification report.
    """
    df = pd.read_csv(dict_file, header=None)
    fam_list = [df[0][i] for i in range(len(df[0]))]

    if std_out:
        r_dict = metrics.classification_report(true_labels, predicted_labels,
                                               target_names=fam_list, digits=3,
                                               output_dict=True)
        # print(r_dict)
        with open(f'{save_dir}/classification_report.csv', 'a') as f:
            print(",precision,recall,f1-score,support", file=f)
            for fam in fam_list:
                print(f"{fam},{r_dict[f'{fam}']['precision']},{r_dict[f'{fam}']['recall']},{r_dict[f'{fam}']['f1-score']},{r_dict[f'{fam}']['support']}",
                      file=f)
            for sm in ['micro avg', 'macro avg', 'weighted avg']:
                print(f"{sm},{r_dict[f'{sm}']['precision']},{r_dict[f'{sm}']['recall']},{r_dict[f'{sm}']['f1-score']},{r_dict[f'{sm}']['support']}",
                      file=f)

    else:
        metrics.classification_report(true_labels, predicted_labels,
                                      target_names=fam_list, digits=3,
                                      output_dict=True)


def k_fold_split_csv(dir_name, filename='train.csv', k=10):
    """Split the origin csv file into k fold csv file.
    """
    # initialization
    seq_list = []
    k_list = []

    with open(f"data/{dir_name}/{filename}", 'r') as f:
        for line in f:
            line = line.strip()
            seq_list.append(line)

    random.shuffle(seq_list)

    num_each = int(len(seq_list) / k)

    # generate a list containing k_fold dataset
    for i in range(k):
        temp_list = seq_list[num_each * i:num_each * (i + 1)]
        k_list.append(temp_list)

    for i in range(k):

        test_list = k_list.pop(0)

        for test_seq in test_list:
            with open(f"data/{dir_name}/test_{i}.csv", 'a') as f:
                print(test_seq, file=f)

        for train_sub_list in k_list:
            for train_seq in train_sub_list:
                with open(f"data/{dir_name}/train_{i}.csv", 'a') as f:
                    print(train_seq, file=f)

        k_list.append(test_list)


if __name__ == "__main__":
    cmd()
