import os
import shutil
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
        default="rfam_mirna",
        help="The directory saving fasta file (default: rfam_mirna)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="The threshold of the maximal length of sequences (default: 200), all longer ones will be filtered out."
    )

    # parser.add_argument(
    #     "--min_length",
    #     type=int,
    #     default=0,
    #     help="The threshold of the minimal length of sequences (default: 0), all shorter ones will be filtered out."
    # )

    parser.add_argument(
        "--split_ratio",
        type=int,
        default=6,
        help="The split ratio of splitting the original dataset into train /test / validation (default: 6)"
    )

    parser.add_argument(
        "--output_dir",
        default="inputdir",
        type=str,
        help="Directory for saving csv (default: inputdir)"
    )

    parser.add_argument(
        "--out_dir_name",
        type=str,
        default=None,
        help="Name of directory saving the results. (default: such as 142_l312_600)"
    )

    parser.add_argument(
        "--mode",
        default=3,
        type=int,
        help="Choose the mode of generating dataset (default: 3). (3: train+validation+test) (2: train+test) (1: test)"
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


def f2c(args, ideal_count, fasta_dir="rfam_mirna", max_l=312, split_ratio=6):
    fasta_list = os.listdir(f"data/{fasta_dir}")
    fasta_list.sort()

    # directory used to save csv file
    if args.out_dir_name is None:
        # csv_dir_name = f"l{args.max_length}_{ideal_count}"
        csv_dir_name = f"{ideal_count}_onehot"
    else:
        csv_dir_name = args.out_dir_name

    # os.system(f"mkdir -p {args.output_dir}/{csv_dir_name}")
    os.mkdir(f"{args.output_dir}/{csv_dir_name}")

    i = 0
    train_fa = []
    test_fa = []
    validation_fa = []

    for fi in fasta_list:
        filename = os.path.join('data', fasta_dir, fi)

        # record the length of each sequence into a list
        length = [len(i) for i in SeqIO.parse(filename, "fasta")]
        length_filter = [val for val in length if (val <= max_l)]

        if len(length_filter) >= ideal_count:
            fam_name = fi.split('.').pop(0).split('_').pop()
            # write a label dictionary ({family: label})
            with open(f"{args.output_dir}/{csv_dir_name}/fam_label.csv", 'a') as f:
                print(f"{fam_name},{i}", file=f)

            print(f"No.{i + 1} family: {fi} have {len(length_filter)} sequences <= {max_l}")

            seq_fasta = [s for s in SeqIO.parse(filename, "fasta") if len(s) <= max_l]

            if args.shuffle:
                random.shuffle(seq_fasta)
            else:
                pass

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
                    with open(f"{args.output_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in validation_seq:
                    with open(f"{args.output_dir}/{csv_dir_name}/validation.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.output_dir}/{csv_dir_name}/test.csv", 'a') as f:
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
                    with open(f"{args.output_dir}/{csv_dir_name}/train.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
                for s in test_seq:
                    with open(f"{args.output_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)

            elif args.mode == 1:
                # save fasta file in a list
                # train_fasta = seq_fasta[:ideal_count]
                if ideal_count == 0:
                    test_fasta = seq_fasta[:]
                else:
                    test_fasta = seq_fasta[:ideal_count]
                for index, s in enumerate(test_fasta):
                    s.id = fam_name + f'_{index}'
                    s.description = ''
                test_fa.extend(test_fasta)

                train_seq = [s.seq for s in test_fasta]

                for s in train_seq:
                    with open(f"{args.data_dir}/{csv_dir_name}/test.csv", 'a') as f:
                        print(f"{i},{s}", file=f)
            i += 1

    # print the results and save the result to the fasta files
    print(f"The training dataset includes {len(train_fa)} sequences.")
    SeqIO.write(train_fa, f"{args.output_dir}/{csv_dir_name}/train.fa", 'fasta')

    if len(test_fa) != 0:
        print(f"The testing dataset includes {len(test_fa)} sequences.")
        SeqIO.write(test_fa, f"{args.output_dir}/{csv_dir_name}/test.fa", 'fasta')

    if len(validation_fa) != 0:
        print(f"The validation dataset includes {len(validation_fa)} sequences.")
        SeqIO.write(validation_fa, f"{args.output_dir}/{csv_dir_name}/validation.fa", 'fasta')
    # rename the output directory: add the number of class
    shutil.move(f"{args.output_dir}/{csv_dir_name}", f"{args.output_dir}/{i}_{csv_dir_name}")


if __name__ == "__main__":
    args = cmd()
    f2c(args, args.num_per_fam, args.fasta_dir, args.max_length, args.split_ratio)
