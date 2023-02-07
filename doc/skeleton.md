## The skeleton code for tiny-voxceleb

We have provided some skeleton code to get you started training and evaluating on the tiny-voxceleb data. Note that the instructions below are intended for running the code on
your own machine. For running the code on the cluster, read [these instructions](cluster.md).

### setting up the environment variables

Make a copy of the `.env.example` file:

```
$ cp .env.example .env
```

And then fill it in accordingly. For example, **on my computer**, I used the following values:

```
SCIENCE_USERNAME=nvaessen
DATA_FOLDER=/home/nvaessen/tiny-voxceleb-skeleton/data
```
You should, of course, use your own username, and point to a directory you own.

### Installing dependencies

You can create a virtual environment with the script

```
$ ./scripts/setup_virtual_environment.sh
```

This should create a folder called `venv` in your current working directory. 

You can then activate the virtual environment so that the `python3` command will find all the third-party libraries.
```
$ source venv/bin/activate
```

If you don't have a GPU on your local computer, you can replace `requirements.txt` with `requirements_cpu.txt` in
`scripts/setup_virtual_environment.sh`.

### Frameworks

The skeleton code uses the following software libraries. In order to fully understand the code we've provided to you, 
it might be beneficial to read some (high-level) documentation, or look up their API references.

1. PyTorch (https://pytorch.org/docs/stable/index.html)
2. TorchAudio (https://pytorch.org/audio/stable/index.html)
3. TorchData (https://pytorch.org/data/main/index.html)
4. PyTorchLightning (https://pytorch-lightning.readthedocs.io/en/latest/)
5. TorchMetrics (https://torchmetrics.readthedocs.io/en/latest/)


### Architecture of skeleton code

Currently, there are 4 submodules under the root `skeleton` python module, with the equivalent folder name `skeleton` in the root of the project.

```
skeleton/
├── data
│   ├── datapipe.py         <-- building blocks for building a torchdata pipeline
│   ├── __init__.py
│   └── tiny_voxceleb.py    <-- train/val/test dataloaders for PyTorch (Lightning)
├── evaluation
│   ├── eer.py           <-- EER metric implementation 
│   ├── evaluation.py    <-- calculation of scores (and EER) from speaker embeddings
│   └── __init__.py
├── __init__.py
├── layers
│   ├── __init__.py
│   └── statistical_pooling.py  <-- custom pooling layers used in network
└── models
    ├── __init__.py
    └── prototype.py    <-- neural network and train/val/test step implementation
```

#### `data` submodule

The `data` submodule contains all the logic related to reading and processing the data. The `tiny_voxceleb.py` file implement the class `TinyVoxcelebDataModule`,
which is a [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html), 
which provides three [PyTorch DataLoaders](https://pytorch.org/docs/stable/data.html#module-torch.utils.data), one each for the train, validation and development data subsets. 
These data loaders are created with several [iterable-style datapipes](https://pytorch.org/data/main/torchdata.datapipes.iter.html)
from TorchData, most importantly [TarArchiveLoader](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.TarArchiveLoader.html#torchdata.datapipes.iter.TarArchiveLoader) 
because reading from
many small files would be a large bottleneck when we train on the cluster. 
The data loaders provide a stream of tuples in the format `(sample_id, audio, ground_truth)`.
There are also several functions to transform the audio, such as chunking into a specific length, batching, and calculating the MFCC.

#### `evaluation` submodule

The `evaluation` submodule contains the logic related to scoring a trial (`evaluation.py`), as well as computing the EER from the resulting scores (`eer.py`).  

#### `layers` submodule

The `layers` submodule is used as an example of how to define custom neural network layers. The basic building block is
called a [Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) in the PyTorch nomenclature.
You can read more about them [here](https://pytorch.org/docs/stable/notes/modules.html). 

In `statistical_pooling.py` we implement two PyTorch modules for a pooling layer which takes a sequence of frames as 
input and returns a fixed-length speaker embedding.

#### `models` submodule

The `models` submodule is used to define the [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)
of the neural network model(s) we are interested in training and evaluating. 
The idea behind a LightningModule is to encapsulate the whole network, as well as how to train and evaluate it, in a single class.
This can be done with a [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) and the `TinyVoxcelebDataModule`.

We provide a `PrototypeSpeakerRecognitionModule` in `prototype.py`, which is a minimal working example of a speaker recognition neural network. 
We have defined the following methods for you:

| method name          | functionality                                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `__init__`           | configures the network layers, hyperparameters, loss function, etc                                                                |
| forward              | Define the inference logic of the model                                                                                           |
| training_step        | Define how to compute the training loss, and other training metrics, based on an input batch                                      |
| training_epoch_end   | Aggregate everything returned by `training_step` over a whole epoch (e.g log training accuracy whole epoch)                       |
| validation_step      | Define how to compute the validation loss, and other validation metrics, based on an input batch                                  |
| validation_epoch_end | Aggregate everything returned by `validation_step` over the whole validation epoch (e.g log validation accuracy over whole epoch) |
| test_step            | Define how to compute the test prediction based on an input batch                                                                 |
| test_epoch_end       | Aggregate everything returned by `test_step` over a whole test epoch, and log test metrics                                        |
| configure_optimizers | Define the PyTorch optimizer(s) and PyTorch LR scheduler(s) used during training                                                  |

Note the distinction between train, val, and test steps:

* In a train step, batches are sampled from the `train` data subset. A loss is computed, which is minimized with the optimizer. 
A train step is done in [train mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train),
which is relevant for e.g., batch norm, and dropout layers.
In a training step additional randomness is often present due to data order shuffling and data augmentation. 
Training metrics are often useful for debugging your network.
* In a val step, batches are sampled from the `val` data subset. 
A loss is computed, but **not** minimized.
Instead, it can be used to visualize when a model starts overfitting. 
A val step is done in [eval mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train). 
A val step does not usually have any sources of randomness (data augmentation is disabled, data order is always the same).
Validation metrics are often used for early stopping, or to select the best model checkpoint during training.
* In a test step, batches are sampled from the `dev` or `eval` data subset. 
A test step is done in [eval mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train). 
A test step does not usually have any sources of randomness (data augmentation is disabled, data order is always the same).
Usually only the relevant metrics to evaluate the model are computed.

The prototype has the following components:

```
PrototypeSpeakerRecognitionModule(
  (embedding_layer): Sequential(
    (0): Conv1d(40, 128, kernel_size=(3,), stride=(1,))
    (1): ReLU()
  )
  (pooling_layer): MeanStatPool1D()
  (prediction_layer): Sequential(
    (0): Linear(in_features=128, out_features=100, bias=True)
    (1): LogSoftmax(dim=1)
  )
  (train_acc): Accuracy()
  (val_acc): Accuracy()
)
```

There is a single CNN layer, which processes the input MFCC. 
Then a pooling layer is used compute a fixed-size speaker embedding. 
Finally, a fully-connected layer is used to compute the train speaker identifies, which we can use with the cross-entropy loss to train the model.
During evaluation, we ignore the fully-connected layer, as we are interested in the speaker embeddings. 

### Training the basic network 

**PLEASE NOTE** For running on the CSEDU compute cluster, you need to submit the commands below as batch jobs, see [the cluster documentation](./cluster.md) for further details.  This section describes the working of the python command line training script.

Make sure the virtual environment is activated before running the command. 
To train the basic network locally, you can use the following:

```
./cli_train.py \
--shard_folder <data>/tiny-voxceleb-shards \
--val_trials_path <data>/tiny-voxceleb/val_trials.txt \
--dev_trials_path <data>/tiny-voxceleb/dev_trials.txt
```

(note that I've inserted the `\ ` and newlines just for readability, they can be omitted in the command line.)

In the above command we assume `<data>` is substituted to the root folder of the data.

There are also already some hyperparameters you can play with:

```
$ ./cli_train.py --help
Usage: cli_train.py [OPTIONS]

Options:
  --shard_folder PATH             path to root folder containing train, val
                                  and dev shard subfolders  [required]
  --val_trials_path PATH          path to .txt file containing val trials
                                  [required]
  --dev_trials_path PATH          path to .txt file containing dev trials
                                  [required]
  --batch_size INTEGER            batch size to use for train and val split
  --audio_length_num_frames INTEGER
                                  The length of each audio file during
                                  training in number of frames. All audio is
                                  16Khz, so 1 second is 16k frames.
  --n_mfcc INTEGER                Number of mfc coefficients to retain
  --embedding_size INTEGER        dimensionality of learned speaker embeddings
  --learning_rate FLOAT           constant learning rate used during training
  --epochs INTEGER                number of epochs to train for
  --use_gpu BOOLEAN               whether to use a gpu
  --random_seed INTEGER           the random seed
  --num_workers INTEGER           number of works to use for data loading
  --help                          Show this message and exit.
```

During a training run information will be stored in `logs/version_xxxxx` folder. To see graphs of training metrics, you can use Tensorboard:

```
$ tensorboard --logdir logs/lightning_logs
```

A checkpoint with the best-performing weights will also be stored under `logs/lightning_logs/version_xxxxx/checkpoints`.

You can look at `experiments/experiment_1_local.sh` for an example script which starts a training run on your local computer.

### Evaluating a network

We have an evaluation script which you can use to compute score lists on the dev and eval test set:

```
$ ./cli_evaluate.py --help
usage: cli_evaluate.py [-h] [--n_mfcc N_MFCC] [--network NETWORK] [--use-gpu USE_GPU] checkpoint shards_dirs_to_evaluate trial_lists score_file

positional arguments:
  checkpoint            path to checkpoint file
  shards_dirs_to_evaluate
                        paths to shard file containing all audio files in the given trial list (',' separated)
  trial_lists           file with list of trials (',' separated)
  score_file            output file to store scores

options:
  -h, --help            show this help message and exit
  --n_mfcc N_MFCC       Number of mfc coefficients to retain, should be the same as in train
  --network NETWORK     which network to load from the given checkpoint
  --use-gpu USE_GPU     whether to evaluate on a GPU device

```

When you've trained your network, you will have a checkpoint to the model in, e.g. 
`logs/lightning_logs/version_x/checkpoints/epoch_0006.step_000001112.val-eer_0.2602.best.ckpt`.
You can then use the following command to generate the score lists:

```
./cli_evaluate.py \
logs/lightning_logs/version_x/checkpoints/epoch_0006.step_000001112.val-eer_0.2602.best.ckpt \
data/tiny-voxceleb-shards/dev/,data/tiny-voxceleb-shards/eval \
data/tiny-voxceleb/dev_trials.txt,data/tiny-voxceleb/eval_trials_no_gt.txt \
scores.txt
```

This will store the scores in a `scores.txt` file. You can use this text file to submit your scores (see [project.md](project.md#submitting-scores)).

Note that when you make changes to the training script, such as a different network, or a new data pipeline, you'll need to adapt the evaluation script as well.
