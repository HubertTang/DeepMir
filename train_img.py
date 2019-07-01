import time, os
import numpy as np
import pandas as pd
from keras.utils import plot_model
from sklearn.metrics import accuracy_score

import utils
import rna_dataset
import rna_model


def train(train_dataset, valid_dataset, test_dataset, fam_path,
          num_channels, num_trains, num_valids, num_tests, args):
    # load model
    model = eval(f"rna_model.{args.model}")(
        seq_length=args.seq_length, num_filters=args.num_filters,
        num_channels=num_channels, filter_sizes=args.filter_sizes, dropout_rate=args.keep_prob,
        num_classes=args.num_classes, num_hidden=args.num_hidden
    )
    print(model.summary())

    # model compile
    model.compile(
        loss=args.loss_function,
        optimizer=args.optimizer,
        metrics=['accuracy']
    )

    # start and record training history
    train_history = model.fit_generator(
        train_dataset,
        epochs=args.num_epochs,
        verbose=1,
        validation_data=valid_dataset,
        workers=6,
        use_multiprocessing=True
    )

    # test accuracy
    t1 = time.time()
    scores = model.evaluate_generator(test_dataset, steps=num_tests // args.batch_size + 1)
    delta_t = time.time() - t1
    print(f"Running time (Prediction):{delta_t} (s)\nAccuracy:{scores[1]}")

    # =================================logging=============================================
    local_time = time.strftime("%m-%d_%H-%M", time.localtime())
    # determine log file name and `mkdir`
    if args.log_name is None:
        log_file_name = local_time
    else:
        log_file_name = local_time + '_' + args.log_name
    os.system(f"mkdir -p {args.log_dir}/{log_file_name}")

    # save model to .h5 file
    model.save(f"{args.log_dir}/{log_file_name}/{log_file_name}.h5")

    # save the image of model structure
    plot_model(model, to_file=f"{args.log_dir}/{log_file_name}/model_structure.png", show_shapes=True)

    # save history to .csv file
    with open(f"{args.log_dir}/history.csv", 'a') as csv:
        print(f'{local_time},{log_file_name},{args.dataset},{scores[1]},{args.model},{args.encode},{str(args.filter_sizes).replace(","," ")},{args.num_filters},{args.batch_size},{args.num_epochs},{args.keep_prob},{args.num_hidden},{args.learning_rate},{args.loss_function},{args.optimizer},{args.remark}',file=csv)

    # save confusion matrix into .csv file
    prediction = model.predict_generator(test_dataset, workers=6, use_multiprocessing=True)
    prediction_1d = np.array([np.argmax(prediction) for prediction in prediction])
    # generate the list of the true label
    label_list = np.zeros((num_tests,), dtype=int)
    no_label = 0
    for i in range(1, num_tests):
        if i % int(num_tests / args.num_classes) == 0:
            no_label += 1
        label_list[i] = no_label

    utils.cm2csv(true_labels=label_list, predicted_labels=prediction_1d,
                 dict_file=fam_path, save_dir=f"{args.log_dir}/{log_file_name}")
    print('CM accuracy:', accuracy_score(label_list, prediction_1d))

    # draw and save history plot
    utils.plot_history(train_history, f"{args.log_dir}/{log_file_name}")

    # generate the confusion matrix
    if args.num_classes <= 15:
        utils.plot_cm(true_labels=label_list, predicted_labels=prediction_1d,
                      dict_file=fam_dict_path, title=f'Confusion Matrix',
                      save_dir=f"{args.log_dir}/{log_file_name}")
    else:
        pass

    # save the classification report
    utils.classification_report(true_labels=label_list, predicted_labels=prediction_1d,
                                dict_file=fam_dict_path, save_dir=f"{args.log_dir}/{log_file_name}",
                                std_out=True)


if __name__ == "__main__":
    args = utils.cmd()
    print(args)

    # generate the dict of dataset path
    partition = {}  # initialize
    labels = {}
    fam_dict = {}
    # data_dir_list = os.listdir(f"data/{args.dataset}")

    for sub_dir in ['train', 'validation', 'test']:
        temp_list = []
        fam_dir_list = os.listdir(f"data/{args.dataset}/{sub_dir}")
        fam_dir_list.sort()
        for la, fam in enumerate(fam_dir_list):
            fam_dict[f"{fam}"] = la
            img_list = os.listdir(f"data/{args.dataset}/{sub_dir}/{fam}")
            for img in img_list:
                temp_list.append(f"{args.dataset}/{sub_dir}/{fam}/{img}")
                labels[f"{args.dataset}/{sub_dir}/{fam}/{img}"] = la
        partition[f"{sub_dir}"] = temp_list

    # check if dataset folder already has `fam_label.csv`
    if 'fam_label.csv' in os.listdir(f"data/{args.dataset}"):
        os.system(f"rm -r data/{args.dataset}/{args.family_dict_file}")
    else:
        pass
    # write a label dictionary ({family: label})
    with open(f"data/{args.dataset}/{args.family_dict_file}", 'a') as f:
        for key, value in fam_dict.items():
            print(f"{key},{value}", file=f)

    # get the paths of family directory
    fam_dict_path = os.path.join("data", args.dataset, args.family_dict_file)

    # get the number of channel
    num_c = {'a': 1, 'b': 1, 'abh': 1, 'bah': 1, 'ab': 3, 'ba': 3}
    cha_c = f"{args.dataset}".split('_').pop()
    n_channels = num_c[f"{cha_c}"]

    # statistic the size of the dataset
    num_trains = len(partition['train'])
    num_validations = len(partition['validation'])
    num_tests = len(partition['test'])
    print(num_trains, num_tests, num_validations)
    # generate dataset
    train_generator = eval(f"rna_dataset.{args.encode}")(partition['train'], labels, batch_size=args.batch_size,
                                                         dim=(args.seq_length, args.seq_length),
                                                         num_channels=n_channels, num_classes=args.num_classes,
                                                         shuffle=True)
    validation_generator = eval(f"rna_dataset.{args.encode}")(partition['validation'], labels,
                                                              batch_size=args.batch_size,
                                                              dim=(args.seq_length, args.seq_length),
                                                              num_channels=n_channels, num_classes=args.num_classes,
                                                              shuffle=True)
    test_generator = eval(f"rna_dataset.{args.encode}")(partition['test'], labels,
                                                        batch_size=int(num_tests / args.num_classes),
                                                        dim=(args.seq_length, args.seq_length),
                                                        num_channels=n_channels, num_classes=args.num_classes,
                                                        shuffle=False)

    train(train_generator, validation_generator, test_generator, fam_dict_path,
          n_channels, num_trains, num_validations, num_tests, args)
