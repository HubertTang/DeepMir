import os
import shutil
import glob
from pathlib import Path
import random
import cv2
import argparse
from Bio import SeqIO
import numpy as np
from tqdm import tqdm


def cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "num_per_fam",
        type=int,
        help="Number of sequences per family (Needs to be divisible by `split_ratio`)"
    )

    parser.add_argument(
        "mat_type",
        type=str,
        help="The type of dataset. (pair, prob, mixed)"
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
    #     help="Minimal length of sequences (default: 0)"
    # )

    parser.add_argument(
        "--split_ratio",
        type=int,
        default=6,
        help="The split ratio of splitting the original dataset into train /test / validation (default: 6)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="The threshold used to generate matrix. (Optional: 0,1,2,3,4) (default: 0)"
    )

    parser.add_argument(
        "--coding",
        type=int,
        default=1,
        help="The pair coding methods (default: 1). (1: ordered) (2: unordered)"
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
        help="Name of output directory name which saving the data. (Add the number of families automaticlly)"
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
    assert (args.mat_type in ['pair', 'prob', 'mixed'])

    return args


random.seed(66)


class f2i(object):
    """Extracting sequence from fasta file, generate dot plot file, finally convert into image.
    """

    def __init__(self, fasta_dir, data_dir, out_dir_name, min_length=0, max_length=312, num_per_fam=600, split_ratio=3):

        self.fasta_dir = fasta_dir
        self.data_dir = data_dir
        self.out_dir_name = out_dir_name
        self.min_length = min_length
        self.max_length = max_length
        self.num_per_fam = num_per_fam
        self.split_ratio = split_ratio

    def extract_seq(self):
        """Extract a specific number of required sequences.

        Returns:
            Save fasta file into `temp_seq` directory.
        """
        # check if current folder already has `temp_seq`
        if 'temp_seq' in os.listdir(os.getcwd()):
            # os.system("rm -r temp_seq")
            shutil.rmtree("temp_seq")
        else:
            pass
        # generate a new directory to save fasta file.
        # os.system("mkdir temp_seq")
        os.makedirs("temp_seq")

        # Extract a specific number of required sequences.
        fasta_dir_list = os.listdir(f"data/{self.fasta_dir}")
        # i = 0   # initialization

        print('Extracting required sequences ... ...')
        for fi in tqdm(fasta_dir_list):
            filename = os.path.join("data", self.fasta_dir, fi)
            length = [len(i) for i in SeqIO.parse(filename, "fasta")]
            length_filter = [val for val in length
                             if (self.min_length <= val <= self.max_length)]

            if len(length_filter) >= self.num_per_fam:
                # family name
                family_name = fi.split("_").pop().split('.').pop(0)

                # extract sequences with specific length
                seq_filter = [s for s in
                              SeqIO.parse(filename, "fasta")
                              if len(s) <= self.max_length]

                if args.shuffle:
                    random.shuffle(seq_filter)
                else:
                    pass
                # if self.num_per_fam == 0:
                #     seq_extra = seq_filter[:]
                # else:
                #     # seq_extra = seq_filter[:self.num_per_fam]
                #     seq_extra = seq_filter[:]
                seq_extra = seq_filter[:]

                if args.mode == 3:
                    split_1 = int(self.num_per_fam / self.split_ratio * (self.split_ratio - 2))
                    split_2 = int(self.num_per_fam / self.split_ratio)
                    train_seq = seq_extra[: split_1]
                    validation_seq = seq_extra[split_1:split_1 + split_2]
                    test_seq = seq_extra[split_1 + split_2:split_1 + split_2 * 2]

                    # create diretory to save fasta file
                    # os.system(f"mkdir -p temp_seq/train/{family_name}")
                    # os.system(f"mkdir -p temp_seq/validation/{family_name}")
                    # os.system(f"mkdir -p temp_seq/test/{family_name}")
                    os.makedirs(f"temp_seq/train/{family_name}")
                    os.makedirs(f"temp_seq/validation/{family_name}")
                    os.makedirs(f"temp_seq/test/{family_name}")

                    # save sequences to the fasta file in the train_dp_path and test_dp_path respectively
                    SeqIO.write(train_seq, f"temp_seq/train/{family_name}/{family_name}.fa", "fasta")
                    SeqIO.write(validation_seq, f"temp_seq/validation/{family_name}/{family_name}.fa", "fasta")
                    SeqIO.write(test_seq, f"temp_seq/test/{family_name}/{family_name}.fa", "fasta")

                if args.mode == 2:
                    split_1 = int(self.num_per_fam / self.split_ratio * (self.split_ratio - 1))
                    split_2 = int(self.num_per_fam / self.split_ratio)
                    train_seq = seq_extra[: split_1]
                    test_seq = seq_extra[split_1:split_1 + split_2]

                    # create diretory to save fasta file
                    # os.system(f"mkdir -p temp_seq/train/{family_name}")
                    # os.system(f"mkdir -p temp_seq/test/{family_name}")
                    os.makedirs(f"temp_seq/train/{family_name}")
                    os.makedirs(f"temp_seq/test/{family_name}")

                    # save sequences to the fasta file in the train_dp_path and test_dp_path respectively
                    SeqIO.write(train_seq, f"temp_seq/train/{family_name}/{family_name}.fa", "fasta")
                    SeqIO.write(test_seq, f"temp_seq/test/{family_name}/{family_name}.fa", "fasta")

                if args.mode == 1:
                    if self.num_per_fam == 0:
                        test_seq = seq_extra[:]
                    else:
                        test_seq = seq_extra[:self.num_per_fam]

                    # create diretory to save fasta file
                    # os.system(f"mkdir -p temp_seq/test/{family_name}")
                    os.makedirs(f"temp_seq/test/{family_name}")

                    # save sequences to the fasta file in the train_dp_path and test_dp_path respectively
                    SeqIO.write(test_seq, f"temp_seq/test/{family_name}/{family_name}.fa", "fasta")

                # print(f"No.{i+1} family: {fi} have {len(length_filter)} satisfied sequences.")

    def generate_dp(self):
        """Generate dotplot file, using fasta files from `temp_seq`

        Returns:
            Directory `temp_dp` containing dot plot files generated by RNAfold.
        """
        self.extract_seq()  # extract fasta file

        # check if current folder already has `temp_dp`
        if 'temp_dp' in os.listdir(os.getcwd()):
            # os.system("rm -r temp_dp")
            shutil.rmtree("temp_dp")
        else:
            pass
        # generate a new directory to save dot plot file.
        # os.system("cp -r temp_seq temp_dp")
        shutil.copytree("temp_seq", "temp_dp")

        dir_list = os.listdir("temp_dp")
        print('Generating dot plot file ... ...')
        for sub_dir in tqdm(dir_list):

            sub_filelist = os.listdir(f"temp_dp/{sub_dir}")
            for fam in tqdm(sub_filelist):
                # os.system(f"cd temp_dp/{sub_dir}/{fam}\nRNAfold -p -o < {fam}.fa\nrm *fold *ss.ps *fa")
                os.system(f"cd temp_dp/{sub_dir}/{fam}\nRNAfold -p -o < {fam}.fa")
                for p in Path(f"temp_dp/{sub_dir}/{fam}").glob("*fold"):
                    p.unlink()
                for p in Path(f"temp_dp/{sub_dir}/{fam}").glob("*ss.ps"):
                    p.unlink()
                for p in Path(f"temp_dp/{sub_dir}/{fam}").glob("*fa"):
                    p.unlink()

    def generate_mat(self, filename, MatrixSize, ProbThreshold, MPair_coding):
        """Function used to generate matrix.
        
        Returns:
            Matrix containing sequence 1st and 2nd structure.
        """
        # initialize the coding methods
        coding_1 = {'au': 42.5, 'AU': 42.5, 'UA': 85, 'ua': 85, 'GC': 127.5, 'gc': 127.5, 'CG': 170, 'cg': 170,
                    'GU': 212.5, 'gu': 212.5, 'UG': 255, 'ug': 255}
        coding_2 = {'au': 85, 'AU': 85, 'UA': 85, 'ua': 85, 'GC': 170, 'gc': 170, 'CG': 170, 'cg': 170,
                    'GU': 255, 'gu': 255, 'UG': 255, 'ug': 255}

        # function used to generate the matrix of the base pair
        def MPair(infolist, MatrixSize, coding):
            if coding == 1:
                PairMap = coding_1
            elif coding == 2:
                PairMap = coding_2

            MatrixPair = np.zeros((MatrixSize, MatrixSize))

            for j in range(len(infolist)):
                coordX = int(infolist[j][0]) - 1
                coordY = int(infolist[j][1]) - 1
                if infolist[j][3] == 'ubox':
                    MatrixPair[coordX][coordY] = PairMap[f'{infolist[j][4]}']
                else:
                    # MatrixPair[coordY][coordX] = 0
                    break

            return MatrixPair

        # function used to generate the matrix of the base pairing probabilities
        def MProb(infolist, MatrixSize):
            MatrixProb = np.zeros((MatrixSize, MatrixSize))

            for j in range(len(infolist)):
                coordX = int(infolist[j][0]) - 1
                coordY = int(infolist[j][1]) - 1
                if infolist[j][3] == 'ubox':
                    MatrixProb[coordX][coordY] = int(255 * float(infolist[j][2]))
                else:
                    # MatrixProb[coordY][coordX] = 0
                    break

            return MatrixProb

        # function used to generate the matrix of the base pairing probabilities(top-half) and base pair(bottle-half)
        def MProbPair(infolist, MatrixSize, coding):
            if coding == 1:
                PairMap = coding_1
            elif coding == 2:
                PairMap = coding_2

            MatrixProbPair = np.zeros((MatrixSize, MatrixSize))

            for j in range(len(infolist)):
                coordX = int(infolist[j][0]) - 1
                coordY = int(infolist[j][1]) - 1
                if infolist[j][3] == 'ubox':
                    MatrixProbPair[coordX][coordY] = int(255 * float(infolist[j][2]))
                    MatrixProbPair[coordY][coordX] = PairMap[f'{infolist[j][4]}']
                else:
                    # MatrixProb[coordY][coordX] = 0
                    break

            return MatrixProbPair

        def dp_parser(filename):
            """Dot plot file parser.

            Returns:
                SeqInf (list): sequence information
            """
            with open(filename) as f:
                file = f.read()
                lines = file.splitlines()

                # initialization
                seq = ''
                re_seq = False
                re_prob = False
                SeqInf = []
                # i = 0
                for i in range(10000000000):
                # while():
                    if re_seq:
                        if lines[i][0] in ['A', 'C', 'G', 'U', 'N', 'T',
                                           'Y', 'R', 'W', 'M', 'K', 'S',
                                           'B', 'H', 'D', 'V', 'X']:
                            seq += str(lines[i].split('\\').pop(0))
                        else:
                            re_seq = False

                    if lines[i].split(' ').pop(0) == '/sequence':
                        re_seq = True

                    if re_prob:
                        if str.isdigit(lines[i][0]):
                            if float(lines[i].split(' ').pop(-2)) >= ProbThreshold:
                                x = int(lines[i].split(' ').pop(0)) - 1
                                y = int(lines[i].split(' ').pop(1)) - 1
                                newStr = lines[i] + f' {seq[x]}{seq[y]}'
                                SeqInf.append(list(newStr.split(' ')))
                        else:
                            break

                    if lines[i] == "%start of base pair probability data":
                        re_prob = True

                    # i += 1

            return SeqInf

        # generate the matrix with 3 channels:
        # channel 0: base pairing probabilities
        # channel 1: base pair
        # channel 2: zeros
        if args.mat_type == 'mixed':
            # def GM_3D_MPair_MProb(filename, MatrixSize, ProbThreshold, MPair_coding):
            matrix = np.zeros((MatrixSize, MatrixSize, 3))

            SeqInf = dp_parser(filename)
            matrix[:, :, 0] = MPair(SeqInf, MatrixSize, MPair_coding)
            matrix[:, :, 1] = MProb(SeqInf, MatrixSize)

        # generate the matrix with 1 channels:
        # channel 0: base pair
        elif args.mat_type == 'pair':
            # def GM_2D_MPair(filename, MatrixSize, ProbThreshold, MPair_coding):

            SeqInf = dp_parser(filename)
            matrix = MPair(SeqInf, MatrixSize, MPair_coding)

        # generate the matrix with 1 channels:
        # channel 0: base pairing probabilities
        elif args.mat_type == 'prob':
            # def GM_2D_MProb(filename, MatrixSize, ProbThreshold, MPair_coding=None):

            SeqInf = dp_parser(filename)
            matrix = MProb(SeqInf, MatrixSize)

        return matrix  # return a (matrix_size, matrix_size) array

    def generate_dataset(self):
        """Generating final dataset using dot plot file in `temp_dp`

        Returns:
            Final images dataset.
        """
        # self.generate_dp()
        # set the threshold(data is based on default threshold = 1e-5)
        # threshold = {0: 0, 1: 0.0042104928, 2: 0.005672006,
        #              3: 0.0077580131999999994, 4: 0.010888888200000002}
        threshold = {0: 0, 1: 0.004869997, 2: 0.0077156579999999985,
                     3: 0.013073067000000002, 4: 0.024114308799999998}

        # check if current folder already has `temp_dp`
        if 'temp_img' in os.listdir(os.getcwd()):
            # os.system("rm -r temp_img")
            shutil.rmtree("temp_img")
        else:
            pass
        # generate a new directory to save fasta file.
        # os.system("mkdir temp_img")
        os.makedirs("temp_img")

        dir_list = os.listdir("temp_dp")
        print("Generating images ... ...")
        for sub_dir in tqdm(dir_list):
            # create directory to save dataset
            # os.system(f"mkdir -p temp_img/{sub_dir}")
            os.makedirs(f"temp_img/{sub_dir}")

            fam_list = os.listdir(f"temp_dp/{sub_dir}")
            for fam in tqdm(fam_list):
                # creat sirectory to save 'train', 'validation', 'test' dataset
                # os.system(f"mkdir -p temp_img/{sub_dir}/{fam}")
                os.makedirs(f"temp_img/{sub_dir}/{fam}")

                dp_file_list = os.listdir(f"temp_dp/{sub_dir}/{fam}")
                ids = 1
                for dp_file in dp_file_list:
                    # print(sub_dir, fam, dp_file)
                    mat = self.generate_mat(filename=f"temp_dp/{sub_dir}/{fam}/{dp_file}",
                                            MatrixSize=args.max_length,
                                            ProbThreshold=threshold[args.threshold],
                                            MPair_coding=args.coding)

                    cv2.imwrite(f"temp_img/{sub_dir}/{fam}/{fam}_{ids:03}.png", mat)
                    ids += 1

        if args.out_dir_name is None:
            out_name = f"{len(fam_list)}_{self.num_per_fam}_{args.mat_type}"
        else:
            out_name = f"{len(fam_list)}_{args.out_dir_name}_{args.mat_type}"

        shutil.move("temp_img", f"{self.data_dir}/{out_name}")
        shutil.move("temp_seq", f"{self.data_dir}/{out_name}/sequence")
        shutil.move("temp_dp", f"{self.data_dir}/{out_name}/dot_plot")


if __name__ == "__main__":
    args = cmd()

    seq = f2i(args.fasta_dir, args.output_dir, args.out_dir_name, 0, args.max_length, args.num_per_fam,
              args.split_ratio)
    seq.generate_dp()
    seq.generate_dataset()
