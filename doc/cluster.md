## Use of the educational GPU cluster

The data science group has a small compute cluster for educational use.  We are going to use this for the Speaker Recognition Challenge of the course [MLiP 2023](https://brightspace.ru.nl/d2l/home/333310).  

The cluster consists of two _compute nodes_, lovingly named `cn47` and `cn48`, and a so-called _head node_, `slurm22`.  All these machines live in the domain `science.ru.nl`, so the head node's fully qualified name is `slurm22.science.ru.nl`.  

Both compute nodes have the following specifications:
 - 8 Nvidia RTX 2080 Ti GPUs, with 11 GB memory
 - 48 Xeon CPUs
 - 128 GB memory, shared between the CPUs
 - Linux Ubuntu 20.04 operating system

The head node has the same OS installed as the compute nodes, but does not have GPUs, and is not intended for heavy computation.  The general idea is that you use the head-node for
 - simple editing and file manipulation
 - submitting jobs to the compute nodes and controlling these jobs

### accessing the cluster

You need a [science account](https://wiki.cncz.science.ru.nl/Nieuwe_studenten#.5BScience_login_.28vachternaam.29_.5D.5BScience_login_.28isurname.29.5D) in order to be able to log into the cluster.  

These nodes are not directly accessible from the internet, in on order to reach these machines you need to either
 - use the science.ru [VPN](https://wiki.cncz.science.ru.nl/Vpn)
   - you have direct access to `slurm22`, this is somewhat easier with copying through `scp` and `rsync`, remote editing, etc.
   - ```
     local+vpn$ ssh $SCIENCE_USERNAME@slurm22.science.ru.nl
     ```
 - login through the machine `lilo.science.ru.nl`
   - The preferred way is to use the `ProxyJump` option of ssh:
        ```
        local$ ssh -J $SCIENCE_USERNAMElilo.science.ru.nl $SCIENCE_USERNAME@slurm22.science.ru.nl
        ```
   - Alternatively, you can login in two steps. In case you have to transport files, please be reminded only your (small) home filesystem `~` is available on `lilo`. 
     ```
       local$ ssh $SCIENCE_USERNAME@lilo.science.ru.nl
       lilo7$ ssh slurm22
     ```

Either way, you will be working through a secure-shell connection, so you must have a `ssh` client on your local laptop/computer.  

### Understanding the filesystems on the cluster

There are several places where you can store code and data. They have different characteristics:

| filesystem           | size        | speed   | scope  |
|----------------------|-------------|---------|--------| 
|  `~`                 | 10 GB       | fast    | shared | 
|  /scratch            | few T       | fastest | local  | 
|  /ceph/csedu-scratch | several TB  | slow    | shared | 

The limitations on the home filesystem, `~` (a.k.a. `$HOME`) are pretty tight---just installing pytorch typically consumes a significant portion of your disk quota.  We have a "[cluster preparation" script](../scripts/prepare_cluster.sh) that will set up an environment for you that will give you best experience working on the cluster:
 - python packages are installed in a virtual environment
 - source data, logs, and models are put on large shared filesystems `/ceph`
 - python libraries are copied to all fast local filesystems `/scratch` 
 - soft-links to these places are made into the project directory
 - the project code is available on fast shared filesystems `~`
 
### Forking and cloning the repository

Before you can carry out the instructions below properly, you need to fork this repository on Gitlab, check out a clone on your home directory on the cluster, and setup the environment. You can follow the [instructions here](./clone.md).

## SLURM

The cluster is an environment where multiple people use computer resources in a co-operative way.  Something needs to manage all these resources, and that process is called a _workload manager_.  At science.ru we use [SLURM](https://slurm.schedmd.com/documentation.html) to do this, like in many other compute clusters in the world.  

Slurm is a clever piece of software, but in the tradition of hard-core computing environments most of the documentation that is available is in plain text "man pages" and inaccessible mailing lists.  View the experience as a time machine, going back to the 1970's...

### Running an interactive SLURM session

It is possible to ask for an interactive shell to one of the compute nodes.  This will only work smoothly if there is a slot available.  If the cluster is "full", jobs will wait until a slot is available, and this may take a while.  An interactive session takes up a slot.  In this example we will ask for a single GPU, the command `srun` is what makes it all happen, the other commands run inside of the session fired up by `srun`:
```
srun --pty --partition csedu --gres gpu:1 /bin/bash
hostname ## we're on cn47 or cn48
nvidia-smi ## it appears there is 1 GPU available in this machine
exit ## make the slot available again, exit to slurm22 again
```
In general, we would advice not to use the interactive shell option, as described here, with a GPU and all, unless you need to just do a quick check in a situation where a GPU is required.  

### Queuing slurm jobs

The normal way of working on the cluster is by submitting a batch job.  This consists of several components:
 - a script (typically bash) that contains all instructions to run the script
 - job control information specifying resources that you need for the job
 - information on where to store the output (standard out and error)

A job is submitted using `sbatch`, specifying the script as an argument and the other information as options.  

As an example, look at [this file](./../experiments/slurm-job.sh), which is a minimalistic script that just gives some information about the environment in which the script runs.  You can submit this for running on the cluster using
```bash
sbatch --partition csedu --gres gpu:1 experiments/slurm-job.sh
squeue
```
The `sbatch` will return immediately (unlike the `srun` earlier) and if you were quick enough with typing the `squeue` you might have seen your job either running or being queued in the job queue.  

When the job has started, you will find a file named `slurm-$jobid.out` in the current working directory:
```bash
ls slurm-*
```
This is where the standard output of the script is collected. 

### More advanced slurm scripts

Having the metadata (`--partiton`, `--gres`, etc) on the command line separate from the script may not always be handy.  Therefore SLURM allows the specification of the job metadata _inside_ the script, by using a special `#SBATCH` syntax.  For bash (and most other script languages) the `#` starts a comment, so it has no meaning to the script itself. 

A full example is in [the skeleton training script](./../experiments/experiment_1_cluster.sh).  Inspect the top of this script, it contains tons of instructions for `sbatch`.  

This skeleton training script is written in a "relative paths" style, assuming you will submit the job while your current working directory is the root of this repository, i.e., trough calling `sbatch experiments/experiment_1_cluster.sh`.  E.g., the logfiles are indicated as `./logs/slurm/%J.out`, the `./logs` refers to the link you've made above setting up the virtual environment.  In this way we don't have to put "hard paths" in the script, which would include your user-specific installation directory, and the script will work for every user. 

The following `#SBATCH` options are in this example:
 - `--partition=csedu`: specifying the subset of all science.ru nodes, we will always be using `csedu`, referring to `cn47` and `cn48`. 
 - `--gres=gpu:1`: we want one GPU
 - `--mem=10G`: we think the job will not use more than 10GB of CPU memory
 - `--cpus-per-task=6`: we want to claim 6 CPUs for this task (mainly for the dataloaders)
 - `--time=6:00:00`: we expect the training to be finished well before 6 hours (wall clock) time.  SLURM will terminate the job if it takes longer...
 - `--output=./logs/slurm/%J.out`: The place were the stdout is collected. `%J` refers to the job ID.  
 - `--error=./logs/slurm/%J.err`: This is where stderr is collected
 - `--mail-type=BEGIN,END,FAIL`: specify that we want a mail message sent to our science account email at the start and finish, and in case of a failed job. 

When you are ready for it, you can run your first [skeleton speaker recognition](./skeleton.md) training job.  The options in the command-line training script are explained [here](./skeleton.md), here we will show you how to submit the job in slurm.  Beware: completing the training takes several hours, even with this [minimalistic neural network](../skeleton/models/prototype.py#L124-126). 

```bash
sbatch experiments/experiment_1_cluster.sh
```
You can now inspect the status of your job using `squeue`, and watch the training progressing slowly using `tail -f logs/slurm/$jobid.out`
