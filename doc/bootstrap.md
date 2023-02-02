## Bootstrapping the first challenge in MLIP

 - Form a team of three or four students
 - Make a [fork](#forking-the-repository-on-scienceru-gitlab) of the repository on Science Gitlab to one of your team member's science account, and add the other team members
 - Configure the Science [VPN](https://wiki.cncz.science.ru.nl/Vpn)
 - Log in to the [compute clusters](cluster.md) machine `cn99`
 - Set up an [ssh private/public key pair](clone.md#etting-up-an-ssh-key-in-order-to-clone-your-copy-of-the-repo) to access this cloned repository from the science cluster
 - [Clone](clone.md#cloning) your private Gitlab repository to the cluster
 - [Set up](clone.md##setting-up-links-and-virtual-environments-in-the-cluster) the environment on the cluster
 - Submit your first [SLURM job](cluster.md#queuing-slurm-jobs)
 - Study the code in the [skeleton](skeleton.md), perhaps make some trivial changes
 - Submit your first speaker recognition [training](skeleton.md#training-the-basic-network) SLURM job with [sbatch](cluster.md#more-advanced-slurm-scripts)
 - Meanwhile, obtain a usename/password for the [leaderboard](https://demo.spraaklab.nl/mlip/2023)
 - Wait for the training to finish
 - [Run inference](skeleton.md#evaluating-a-network), i.e., compute the trial scores for dev and eval data
 - [Make your first submission](project.md#submitting-scores) to the leaderboard system
 - Continue improving your result by adapting the existing code
