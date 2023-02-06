## Honour code

In order to keep this project fun but also educational, we ask you to respect the following rules:

1. You will not develop or publish your code in a publicly-accessible repository (or share it in another manner with peers outside your group) 
2. You will not use a readily-accessible pre-trained network for speaker recognition as your solution.
3. You will not try to find out the identities of the speakers in the evaluation trials from other resources
4. You will not blindly copy code from the internet. You should aim to develop a solution yourself instead of using a code or a library from a third-party. Reading open-source code for inspiration is always allowed!
5. You will respect the SLURM usage etiquette described below.
6. You will not try to cheat the system, e.g., by hacking into the submission server or by doing digital forensics on files and repositories
7. The data provided to you can **only** be used to participate in this Machine Learning in Practise course project.

## Etiquette for using the CSEDU GPU cluster with SLURM

We want every group to be able to use GPU resources provided in the CSEDU compute cluster. Therefore, we ask everyone to honour these rules:

* Do **not** `ssh` into `cn47` and/or `cn48` directly to run any experiments. This leads to other programs crashing due to e.g. out of memory errors, and this way the resources on the cluster cannot be allocated (fairly).
* Your group should only have one running job at a time.
  * If you submit an [array job](https://slurm.schedmd.com/job_array.html), you must use a parallelism of `1`, by using `%1` in e.g. `#SBATCH --array=0-4%1`.
* Your jobs can use a maximum of 6 CPUs, 16 GB memory, and 1 GPU.
  * This can be controlled with the `SBATCH` parameters below 
    ```
    #SBATCH --gres=gpu:1       # this value may not exceed 1
    #SBATCH --mem=10G          # this value may not exceed 16
    #SBATCH --cpus-per-task=6  # this value may not exceed 6
    ```
* Your jobs time-out after at most 24 hours. However, we ask everyone to **aim** for a maximum of 12 hours for most jobs.
  * This can be controlled with `#SBATCH --time=12:00:00`
  * If you have evidence that you need to train for longer than 12 hours, be fair, and restrict your usage afterwards.
  * If you train for longer than 12 hours, make sure that you can argue why this was necessary.
* Use sharded data loading (as implemented in [TinyVoxcelebDataModule](../skeleton/data/tiny_voxceleb.py)), rather than individual file access, wherever you can, to prevent high i/o loads on the network file system.
* Do not run any long-running foreground tasks on the `slurm22` head node.
  * The `slurm22` node should only be used to schedule SLURM jobs
  * An example of short-running foreground tasks with are OK to run on `slurm22`: manipulation of file-system with `rsync` or `cp`, using `git`, using `srun` or `sbatch`.
  * Example of tasks with which should be submitted as a job: offline data augmentation, compiling a large software project.
* Whenever you're using the cluster, use your judgement to make sure that everyone can have access.

## Other rules related to proper evaluation

See the [speaker recognition](speaker-recognition.md) documentation for an explanation of the terms used in the following rules

1. The score for a trial can only be based upon the training, development and validation data, and the audio files mentioned in the trial.  Specifically, using any other audio files in the evaluation data other than those in the trial is not allowed.  This implies that normalization using other evaluation trial scores is not allowed.  
2. The use of manually produced transcripts of the evaluation data is not allowed. 
3. Listening to the evaluation data, or any other form of human interaction with this data, is not allowed.  

