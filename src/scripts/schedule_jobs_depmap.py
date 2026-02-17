import sys

wd_path = "/home/dk538/rds/hpc-work/pico/src"
sys.path.append(wd_path)

import os
import time

import argparse

feat_sets_dict_transneo = {
    "Rep": "",
    "Clinical+Rep": "Size.at.diagnosis LN.at.diagnosis Age.at.diagnosis Histology ER.status HER2.status Grade.pre.chemotherapy",
    "Clinical": "Size.at.diagnosis LN.at.diagnosis Age.at.diagnosis Histology ER.status HER2.status Grade.pre.chemotherapy",
    "Clinical+Rep+RNA": "Size.at.diagnosis LN.at.diagnosis Age.at.diagnosis Histology ER.status HER2.status Grade.pre.chemotherapy PGR.log2.tpm ESR1.log2.tpm ERBB2.log2.tpm GGI.ssgsea.notnorm ESC.ssgsea.notnorm Swanton.PaclitaxelScore STAT1.ssgsea.notnorm TIDE.Dysfunction TIDE.Exclusion Danaher.Mast.cells CytScore.log2",
    "Clinical+RNA": "Size.at.diagnosis LN.at.diagnosis Age.at.diagnosis Histology ER.status HER2.status Grade.pre.chemotherapy PGR.log2.tpm ESR1.log2.tpm ERBB2.log2.tpm GGI.ssgsea.notnorm ESC.ssgsea.notnorm Swanton.PaclitaxelScore STAT1.ssgsea.notnorm TIDE.Dysfunction TIDE.Exclusion Danaher.Mast.cells CytScore.log2",
    "RNA": "PGR.log2.tpm ESR1.log2.tpm ERBB2.log2.tpm GGI.ssgsea.notnorm ESC.ssgsea.notnorm Swanton.PaclitaxelScore STAT1.ssgsea.notnorm TIDE.Dysfunction TIDE.Exclusion Danaher.Mast.cells CytScore.log2",
}

feat_sets_dict_scanb = {
    "Rep": "",
    "Clinical+Rep": "AGE ER HER2 LN SIZE",
    "Clinical": "AGE ER HER2 LN SIZE",
}


def submission_pico_sk_gdsc(
    dataset,
    reg,
    enc,
    target,
    seed,
    experiment=None,
    constraints=None,
    walltime="02:00:00",
):
    if constraints is not None:
        constraints_str = " ".join(constraints)

    submission_preamble = rf"""#!/bin/bash
#SBATCH -J dk538-ssvae
#SBATCH -A MRC-BSU2-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={walltime}
#SBATCH --output=/home/dk538/rds/hpc-work/graphdep/slurm_out/pico_hopt/%j.out
#SBATCH --no-requeue
#SBATCH -p icelake

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
source /home/dk538/.bashrc
module load hdf5/1.8.2
conda activate slurm-torch-2

application="python -u ./scripts/pico_sk_hopt.py"

workdir="$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
"""

    submission_post = r"""cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

eval $CMD"""

    script_args = rf"""CMD="$application -reg {reg} -enc {enc} -target {target} -dataset {dataset} -seed {seed} --newstudy"""
    if constraints is not None:
        script_args = rf"""{script_args} -constraints {constraints_str}"""
    if experiment is not None:
        script_args = rf"""{script_args} --experiment {experiment}"""
    # Don't run on GPU
    # else:
    # script_args = fr'''{script_args} --cuda'''
    script_args = rf'''{script_args}"'''

    submission = rf"""{submission_preamble}
{script_args}
{submission_post}"""

    return submission


def main(args):
    regs = ["ElasticNet", "SVR", "RandomForestRegressor"]
    encs = ["iCoVAE", "VAE"]
    seeds = [4563]

    for seed in seeds:
        for reg in regs:
            for enc in encs:
                # Added  for new model version
                job_path_root = f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/pico/{reg}_{enc.lower()}"
                job_path = f"{job_path_root}/test_metrics.csv"
                # job_path_val = f"{job_path_root}/z_pred_val_0_best_s{seed}.csv"

                if os.path.exists(job_path) and not args.newstudy:
                    print(f"Job previously completed: {job_path}")
                else:
                    if args.dataset == "depmap_gdsc":
                        script = submission_pico_sk_gdsc(
                            dataset=args.dataset,
                            reg=reg,
                            enc=enc,
                            target=args.target,
                            constraints=args.constraints,
                            seed=seed,
                            experiment=args.experiment,
                            walltime=args.walltime,
                        )
                    if args.dataset == "depmap_ctrp":
                        script = submission_pico_sk_gdsc(
                            dataset=args.dataset,
                            reg=reg,
                            enc=enc,
                            target=args.target,
                            constraints=args.constraints,
                            seed=seed,
                            experiment=args.experiment,
                            walltime=args.walltime,
                        )

                    script_name = f"{wd_path}/submissions/submit_{args.dataset}_{reg}_{enc}_{args.target}_{args.experiment}_{seed}"
                    print("Script name: " + script_name)

                    f = open(script_name, "w")
                    f.write(script)
                    f.close()

                    print("Running: sbatch " + script_name)
                    os.system("sbatch " + script_name)
                    time.sleep(0.1)


# To cancel these jobs, run the below
# squeue --me --states=ALL --Format=jobid,name --noheader |
#   grep dk538-ssvae |
#   awk '{print $1}' |
#   xargs scancel


def parser_args(parser):
    parser.add_argument(
        "-target",
        type=str,
        default="ABCD",
        metavar="D",
        help="Target column from y",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="depmap_gdsc",
        metavar="S",
        help="Dataset name (e.g. depmap_gdsc)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        type=str,
        help="Experiment to run (user defined in data loading class)",
    )
    parser.add_argument(
        "-constraints",
        default=None,
        type=str,
        nargs="+",
        help="If specified, uses encoder with these constraints rather than using defaults/selecting automatically",
    )
    parser.add_argument(
        "-walltime",
        default="06:00:00",
        type=str,
        help="Walltime per job",
    )
    parser.add_argument(
        "--newstudy",
        default=False,
        action="store_true",
        help="If specified, resubmits all jobs regardless of previous completion",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    main(args)
