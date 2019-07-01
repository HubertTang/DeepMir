import time, os
import numpy as np
import pandas as pd
from keras.utils import plot_model
from keras import optimizers
from sklearn.metrics import accuracy_score
import utils
import rna_dataset
import rna_model


def train(train_dataset, valid_dataset, validation_bool, test_dataset, fam_dict_path,
          num_column, num_trains, num_tests, test_file_path, args):
    # load model
    model = eval(f"rna_model.{args.model}")(
        seq_length=args.seq_length, num_c=num_column, num_filters=args.num_filters,
        filter_sizes=args.filter_sizes, dropout_rate=args.keep_prob,
        num_classes=args.num_classes, num_hidden=args.num_hidden
    )
    print(model.summary())

    # model compile
    model.compile(
        loss=args.loss_function,
        optimizer=eval(f"optimizers.{args.optimizer}")(lr=args.learning_rate),
        metrics=['accuracy']
    )

    # start and record training history
    if validation_bool:
        train_history = model.fit_generator(
            train_dataset,
            epochs=args.num_epochs,
            verbose=1,
            validation_data=valid_dataset,
            use_multiprocessing=True,
            workers=6
        )
    else:
        train_history = model.fit_generator(
            train_dataset,
            epochs=args.num_epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=6
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
    # prediction = model.predict_generator(test_generator, workers=6, use_multiprocessing=True)
    prediction = model.predict_generator(test_generator)    # don't use the multiprocessing

    # get the list of true label
    with open(test_file_path) as f:
        label_list = []
        for line in f:
            line = line.strip()
            seq_index = line.split(',').pop(0)
            if seq_index != '':
                label_list.append(int(seq_index))
            else:
                pass

    prediction = prediction[:len(label_list)]
    prediction_1d = np.array([np.argmax(prediction) for prediction in prediction])
    # print("Length of true label:", len(label_list))
    # print("Length of predict label:", len(prediction_1d))
    utils.cm2csv(true_labels=label_list, predicted_labels=prediction_1d,
                 dict_file=fam_dict_path, save_dir=f"{args.log_dir}/{log_file_name}")
    print('CM accuracy:', accuracy_score(label_list, prediction_1d))

    # generate the confusion matrix
    if args.num_classes <= 150:
        utils.plot_cm(true_labels=label_list, predicted_labels=prediction_1d,
                      dict_file=fam_dict_path, title=f'Confusion Matrix',
                      save_dir=f"{args.log_dir}/{log_file_name}")
    else:
        pass

    # draw and save history plot
    utils.plot_history(train_history, f"{args.log_dir}/{log_file_name}")

    # save the classification report
    utils.classification_report(true_labels=label_list, predicted_labels=prediction_1d,
                                dict_file=fam_dict_path, save_dir=f"{args.log_dir}/{log_file_name}",
                                std_out=True)


if __name__ == "__main__":
    args = utils.cmd()
    print(args)

    # initialize a variable to test the existence of the validation dataset
    val_bool = True

    # get the paths of each dataset
    # statistic the size of the dataset
    train_path = os.path.join("data", args.dataset, args.train_file)    # training dataset
    num_trains = utils.count_lines(train_path)

    try:    # validation dataset
        valid_path = os.path.join("data", args.dataset, args.valid_file)
        num_validations = utils.count_lines(valid_path)
    except FileNotFoundError:
        val_bool = False

    test_path = os.path.join("data", args.dataset, args.test_file)  # testing dataset
    num_tests = utils.count_lines(test_path)

    fam_dict_path = os.path.join("data", args.dataset, args.family_dict_file)

    # get another dimension
    ano_dim = 0
    if args.encode == "RNA_onehot":
        ano_dim = 4
    elif args.encode in ['RNA_fimg', 'RNA_pimg']:
        ano_dim = args.seq_length
    elif args.encode.split('#').pop(0) == 'RNA_word':
        ano_dim = 5 ** int(args.encode.split('#').pop(1))
    else:
        pass

    # generate dataset
    train_generator = eval(f"rna_dataset.{args.encode}")(train_path, num_trains, batch_size=args.batch_size,
                                                         dim=(args.seq_length, ano_dim),
                                                         num_channels=1, num_classes=args.num_classes, shuffle=True)
    if val_bool:
        validation_generator = eval(f"rna_dataset.{args.encode}")(valid_path, num_validations, batch_size=args.batch_size,
                                                                  dim=(args.seq_length, ano_dim),
                                                                  num_channels=1, num_classes=args.num_classes,
                                                                  shuffle=True)
    else:
        validation_generator = None

    test_generator = eval(f"rna_dataset.{args.encode}")(test_path, num_tests,
                                                        batch_size=args.batch_size,
                                                        dim=(args.seq_length, ano_dim),
                                                        num_channels=1, num_classes=args.num_classes, shuffle=False)

    train(train_generator, validation_generator, val_bool, test_generator, fam_dict_path,
          ano_dim, num_trains, num_tests, test_path, args)
