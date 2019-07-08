# Fast and accurate microRNA search using CNN



# Requirements

* Python: 3.6
* Keras: 2.2.4
* viennarna: 2.4.11
* Other python modules: numpy, matplotlib, scikit-learn, pandas, biopython, opencv, tqdm



# Usage

> You can see the valid parameters of the script by help option:

```bash
$ python <SCRIPT_FILE> --help
```



* Clone the repository or download compressed source code file.

```bash
$ git clone https://github.com/HubertTang/DeepMir
$ cd DeepMir
```

* Generate input file for CNN:

  ***INPUT:*** The directory containing the original `.fasta` files of different families should be saved in `data/`:

  ```bash
  data/
  |---rfam_mirna/
		|---RF00027.fa
  	|---RF00047.fa
		|---RF00051.fa
  		...
  ```
  
  ***OUTPUT:*** The directory containing generated files will be save in `inputdir/` by default. The default name of the directory is like: `165_60_onehot` . `165` is the number of the classes in this dataset, `60` is the number of sequences of each family, `onehot` is the encoding method.
  
  * For pair/ probability/ mixed matrix method:

  > One example of parameter setting for generating `Rfam-60` using pair matrix:

  ```bash
  $ python fasta2img.py 60 pair \
    --fasta_dir rfam_mirna \
    --max_length 200 \
    --split_ratio 6 \
    --threshold 0 \
    --coding 1 \
    --output_dir inputdir \
    --mode 2 \
    --shuffle True
  ```
  
  * For one-hot matrix method:
  
  > One example of parameter setting for generating `Rfam-60` using one-hot matrix:
  
  ```bash
  $ python fasta2onehot.py 60 \
    --fasta_dir rfam_mirna \
    --max_length 200 \
    --split_ratio 6 \
    --output_dir inputdir \
    --mode 2 \
    --shuffle True
  ```

* Train CNN model and evaluate the performance:

  ***INPUT:*** The input file saved in `inputdir/`.

  ***OUTPUT:*** After training, the performance of the model will be evaluated automatically using the testing data. The trained model and the performance will be saved in a directory, named by data, in `log/`.

  Use `train_img.py` to train the pair/ probability/ mixed matrix model, use `train_onehot.py` to train the one-hot matrix model.

  > One example of parameter setting for training pair matrix model using dataset from `165_60_pair`

  ```bash
  $ python train_img.py 165_60_pair 165 \
    --seq_length 200 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --num_epochs 20 \
    --filter_sizes 2 2 \
    --num_filters 32 64 \
    --keep_prob 0.5 \
    --num_hidden 128 64 \
    --loss_function categorical_crossentropy \
    --optimizer Adam \
    --log_dir log
  ```

  

# Data

All datasets can be download from [HERE](https://1drv.ms/u/s!AutVZjbvRAR7i_IKbf-ef32QmifEdw?e=qb9CXP). Please read the `README.txt` before using.



# Contact

If you have any question or problem, please feel free to send email to xubotang2-c@ad.cityu.edu.hk