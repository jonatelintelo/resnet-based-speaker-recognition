## Project details

You will be given a small subset of the Voxceleb dataset. This dataset contains 110 unique speakers, and each speaker has one or more audio recordings extracted from YouTube videos. Based on this data, your task will be to implement a system which learns to distinguish whether two audio recordings belong to the same speaker or to different speakers. 

## Time schedule

TODO update these values
 - 30 January 2023 First lecture, start forming of teams, formal start of challenge
 - 6 February 2023 [Tutorial](bootstrap.md), introducing the skeleton code and more
 - 17 March 2023, 23:59, Deadline for result submissions, disclosure of evaluation results 
 - 21 March 2023 Deadline for submitting report and code 

## Evaluation data

In order to evaluate the system an evaluation set containing audio recordings of a number of *never-seen-before* speakers will be used at the end of the project, after the submission deadline has passed. You will be given a list of trials between two audio recordings of these speakers, and for each trial your model should give a score that is lower, e.g., tending to 0, if the speakers are different and higher, e.g., tending to 1, if the speakers are equal.  See the [section on speaker recognition](speaker-recognition.md) for what a _trial_ is, and how speaker recognition systems are evaluated.


The data provided to you is split up as follows:
* train: audio files from 100 speakers. 50 male, 50 female. The number of audio files and number of recording sources differ per speaker.
* val: audio files from 2 recording sources held-out from the training split, for all 100 training speakers. A trial list of length 10000 over these audio files.
* dev: audio recordings from 10 'unseen' speakers in the train/val split. 5 male, 5 female. A trial list of length 10_000 over these audio files.
* eval: unlabeled audio recordings from unseen speakers which are only used for at the end of the project for a definitive evaluation. A trial list of length 37611 over these audio files.

## Where to find the data

You can readily access the data on the [cluster](cluster.md). It is located at `/ceph/csedu-scratch/course/IMC030_MLIP/data` and structured as follows: 

```
/ceph/csedu-scratch/course/IMC030_MLIP/data/
├── tiny-voxceleb
│   ├── dev
│   ├── eval
│   ├── train
│   ├── val
│   ├── dev_trials.txt
│   ├── eval_trials_no_gt.txt
│   ├── tiny_meta.csv
│   └── val_trials.txt
├── tiny-voxceleb-shards
│   ├── dev
│   ├── eval
│   ├── train
│   └── val
└── data.zip
```

The `tiny-voxcceleb` subfolders contains all audio files in the respective train, val, dev, and eval sets as separate `.wav` files.
It also contains the trials for the validation and dev sets (with ground-truth labels), as well as the trials for the eval set (without ground-truth labels).
The `tiny_meta.csv` file contains some meta information about each speaker (gender, nationality, amount of data, which set they are in, and a classification index)

The `tiny-voxceleb-shards` folder contains the respective audio files in `.tar` files. This is required for training on the cluster as randomly-accessing a lot of small files on a network-attached storage system is very slow.

If you want to play with the data on the local computer, or do some training on your own computer, you can use `scripts/download_data.sh` to download it the zip file to your local `$DATA_FOLDER` folder, where you can then extract it.
The `data.zip` file contains the two folders `tiny-voxceleb` and `tiny-voxceleb-shards`.

## Submitting scores

Once you have a system that can carry out speaker recognition in some way, you should prepare a score file that you can submit to the leaderboard system. 

A score file is a text file with three columns, separated by whitespace.  Each line represents a processed trial.  The first column is the _score_ of the speaker comparison, this is a floating point number.  The second and third columns are the IDs of the audio files that were compared, in the same format as in the trial lists. 

For a valid submission, you need to prepare a file containing scores for all trials from the _dev_ and _eval_ trial lists, that is 10000 scores from `dev_trials.txt` and 37611 score from `eval_trials_no_gt.txt`.  The _dev_ trials are used for the leaderboard before the deadline, the _eval_ scores are stored in the leaderboard system, and will be used to determine the final leaderboard ranking after the submission deadline has passed. 

Note that you have the ground truth of the _dev_ trials, so don't fool yourself by using this ground truth information in your dev-scores!  You could obtain a high leaderboard rank before the deadline. 

An example of the submission format is:
```txt
0.8245347142219543 id10775/DYjmBHx5TlI/00007 id11171/4yT7m_Swm88/00001
0.8159198760986328 id10147/QVipPwhh1GQ/00003 id10147/Q3zL1cvyMfo/00053
0.8003206253051758 id10219/PBou1uIhLYs/00005 id11171/ZlHf-ubfDJU/00007
...
0.8052331209182739 0a684aa17e8446879de5645679c9f4d3 aed6b8508c6643da88158ea5280a860b
0.8998304605484009 af90f303e34f4354b4d97581e68e74c0 7215f829b8124f6ca26cd9c88638faa6
0.8717533349990845 a70a5ad9bc854bb2b5accb19cb7ff50f bb7068b920d34f56a425d6c577f62241
```

The top lines are trials from the dev set, the bottom lines are trials from the eval set.  

Submitting scores to the leaderboard system can be done using:
```bash
./leaderboard.py --team $team --password $password --submit $scorefile [--notes $notes]
```
where `$team` and `$password` are the team identifier and password that will be distributed to you.  You are free to put these values as defaults in the script.  The `$notes` are optional, but it is easier to keep track of your personal results if you tag them with a short note.  Don't forget to use quotation marks around your notes if it is more than one word!

## Leaderboard

The [Leaderboard](https://demo.spraaklab.nl/mlip/2023) is a web frontend to the leaderboard.  You can login to the site with the password described above.  On the site, you can view the current leaderboard, as well as you own submissions.  You can also use a web form to upload a new submission.  

Up to the submission deadline in March, the leaderboard code will show you team's best dev set EER.  After the submission deadline, the evaluation scores will be shown. 

