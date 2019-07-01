import os
import random
import argparse
from Bio import SeqIO


def cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "num_per_fam",
        type=int,
        help="Number of sequences per family \
                (Needs to be divisible by `split_ratio`)"
    )

    parser.add_argument(
        "--fasta_dir",
        type=str,
        default="clean_fasta_files",
        help="Fasta Directory (default: clean_fasta_files)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=312,
        help="Maximal length of sequences (default: 312)"  # 95-percentile length in Rfam dataset
    )

    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimal length of sequences (default: 0)"
    )

    parser.add_argument(
        "--split_ratio",
        type=int,
        default=3,
        help="Maximal length of sequences (default: 6)"
    )

    parser.add_argument(
        "--data_dir",
        default="../data",
        type=str,
        help="Directory for saving csv (default: ../data)"
    )

    parser.add_argument(
        "--out_dir_name",
        type=str,
        default=None,
        help="Name of output directory name which saving the data. (default: such as 142_l312_600)"
    )

    parser.add_argument(
        "--mode",
        default=3,
        type=int,
        help="Choose the mode of generating dataset. (3: train+validation+test) \
                (2: train+test) (1: train) (default: 3)"
    )

    parser.add_argument(
        "--shuffle",
        default=True,
        type=bool,
        help="Shuffle the sequences or not (default: True)"
    )

    args = parser.parse_args()

    # check validity
    assert (args.num_per_fam % args.split_ratio == 0)

    return args


random.seed(66)


def f2c(args, ideal_count, fasta_dir="clean_fasta_files", min_l=0, max_l=312, split_ratio=6):
    fasta_list = os.listdir(fasta_dir)
    fasta_list.sort()

    # directory used to save csv file
    if args.out_dir_name is None:
        if min_l == 0:
            csv_dir_name = f"l{args.max_length}_{ideal_count}"
        else:
            csv_dir_name = f"m{min_l}_l{args.max_length}_{ideal_count}"
    else:
        csv_dir_name = args.out_dir_name

    os.system(f"mkdir -p {args.data_dir}/{csv_dir_name}")

    i = 0
    train_fa = []
    test_fa = []
    validation_fa = []

    for fi in fasta_list:
        filename = os.path.join(fasta_dir, fi)

        # record the length of each sequence into a list
        length = [len(i) for i in SeqIO.parse(filename, "fasta")]
        length_filter = [val for val in length
                         if (min_l <= val <= max_l)]

        if len(length_filter) >= ideal_count:
            fam_name = fi.split('.').pop(0).split('_').pop()
            # write a label dictionary ({family: label})
            with open(f"{args.data_dir}/{csv_dir_name}/fam_label.csv", 'a') as f:
                print(f"{fam_name},{i}", file=f)

            print(f"No.{i + 1} family: {fi} have {len(length_filter)} sequences >={min_l} and <= {max_l}")

            seq_fasta = [s for s in SeqIO.parse(filename, "fasta")
                          if len(s) <= max_l]

            if args.shuffle:
                random.shuffle(seq_fasta)
            else:
                pass
            
            # seq_filter = [s.seq for s in seq_fasta]

            if args.mode == 3:
                split_1 = int(ideal_count / split_ratio * (split_ratio - 2))
                split_2 = int(ideal_count / split_ratio)

                # save fasta file in a list
                train_fasta = seq_fasta[: split_1]
                validation_fasta = seq_fasta[split_1:split_1 + split_2]
                test_fasta = seq_fasta[split_1 + split_2:split_1 + split_2 * 2]

                for index, s in enumerate(train_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                for index, s in enumerate(validation_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                for index, s in enumerate(test_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                train_fa.extend(train_fasta)
                validation_fa.extend(validation_fasta)
                test_fa.extend(test_fasta)

                train_seq = [s.seq for s in train_fasta]
                validation_seq = [s.seq for s in validation_fasta]
                test_seq = [s.seq for s in test_fasta]

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in validation_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/validation.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)

            elif args.mode == 2:
                split_1 = int(ideal_count / split_ratio * (split_ratio - 1))
                split_2 = int(ideal_count / split_ratio)
                
                # save fasta file in a list
                train_fasta = seq_fasta[: split_1]
                test_fasta = seq_fasta[split_1:split_1 + split_2]
                for index, s in enumerate(train_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                for index, s in enumerate(test_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                train_fa.extend(train_fasta)
                test_fa.extend(test_fasta)

                train_seq = [s.seq for s in train_fasta]
                test_seq = [s.seq for s in test_fasta]

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)

            elif args.mode == 1:
                # save fasta file in a list
                # train_fasta = seq_fasta[:ideal_count]
                if ideal_count == 0:
                    train_fasta = seq_fasta[:]
                else:
                    train_fasta = seq_fasta[:ideal_count]
                for index, s in enumerate(train_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                train_fa.extend(train_fasta)

                train_seq = [s.seq for s in train_fasta]

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
            
            elif args.mode == 0:
                # extract specific number of sequences to test
                # rest sequeces is used to train
                # train_fasta = seq_fasta[int(ideal_count/2):]
                # test_fasta = seq_fasta[:int(ideal_count/2)]
                train_fasta = seq_fasta[10:]
                test_fasta = seq_fasta[:10]

                for index, s in enumerate(train_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                for index, s in enumerate(test_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                train_fa.extend(train_fasta)
                test_fa.extend(test_fasta)
                
                # save the sequences into a list
                train_seq = [s.seq for s in train_fasta]
                test_seq = [s.seq for s in test_fasta]

                # save the number of sequences into a csv file
                num_train = len(train_seq)
                num_test = len(test_seq)
                with open(f"{args.data_dir}/{csv_dir_name}/size_dataset.csv", 'a') as f:
                    print(f'{fam_name},{num_train},{num_test}', file=f)

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)

            elif args.mode == 4:
                # extract specific number of sequences to train and test
                # rest sequeces is used to test the z-value
                split_1 = int(ideal_count / split_ratio * (split_ratio - 1))
                split_2 = int(ideal_count / split_ratio)

                # save fasta file in a list
                train_fasta = seq_fasta[: split_1]
                test_fasta = seq_fasta[split_1:split_1 + split_2]
                # posi_fasta = seq_fasta[split_1 + split_2:]
                validation_fasta = test_fasta

                for index, s in enumerate(train_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                for index, s in enumerate(test_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                train_fa.extend(train_fasta)
                test_fa.extend(test_fasta)

                # save the sequences into a list
                train_seq = [s.seq for s in train_fasta]
                test_seq = [s.seq for s in test_fasta]
                validation_seq = [s.seq for s in validation_fasta]

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in validation_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/validation.csv", 'a') as f:
                        print(f"{i},{s}", file=f)

            i += 1

    # print the results and save the result to the fasta files
    print(len(train_fa))
    SeqIO.write(train_fa, f"{args.data_dir}/{csv_dir_name}/train.fa", 'fasta')

    if len(test_fa) != 0:
        print(len(test_fa))
        SeqIO.write(test_fa, f"{args.data_dir}/{csv_dir_name}/test.fa", 'fasta')

    if len(validation_fa) != 0:
        print(len(validation_fa))
        SeqIO.write(validation_fa, f"{args.data_dir}/{csv_dir_name}/validation.fa", 'fasta')
    # rename the output directory: add the number of class
    os.system(f"mv {args.data_dir}/{csv_dir_name} {args.data_dir}/{i}_{csv_dir_name}")


if __name__ == "__main__":
    args = cmd()
    f2c(args, args.num_per_fam, args.fasta_dir, args.min_length, args.max_length,
        args.split_ratio)
