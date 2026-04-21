"""Generate and submit SLURM jobs for PiCoSK on DepMap (GDSC / CTRP) drug targets.

This script writes a SLURM submission script for each combination of regressor
('ElasticNet', 'SVR', 'RandomForestRegressor'), encoder ('iCoVAE', 'VAE'), and
seed (4563), then submits each job via `sbatch`. Submitted jobs invoke
src/scripts/pico_sk_hopt.py with the matching command-line arguments. The
dataset (depmap_gdsc or depmap_ctrp) selects which submission template to use
(both currently share the same template in this script).

Inputs are command-line arguments listing the dataset, target drug, optional
experiment tag, constraints, walltime and a --newstudy resubmission flag.
Outputs are SLURM scripts written to <wd_path>/submissions/ and the queued
jobs themselves; if a target output already contains test_metrics.csv the job
is skipped unless --newstudy is passed.

Example:
    python src/scripts/schedule_jobs_depmap.py -dataset depmap_gdsc \\
        -target AFATINIB --experiment h16 -walltime 06:00:00
"""

import argparse
import os
import sys
import time

wd_path = os.environ.get(
    "PICO_SRC",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.append(wd_path)

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
#SBATCH -J pico_hopt                # EDIT: pick a job name
#SBATCH -A <your-slurm-account>     # EDIT: replace with your project/account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={walltime}
#SBATCH --output=slurm_out/pico_hopt/%j.out   # EDIT: log output path
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
source ~/.bashrc
module load hdf5/1.8.2
conda activate pico                 # EDIT: name of the conda env where pico is installed

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
                job_path_root = f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/pico/{reg}_{enc.lower()}"
                job_path = f"{job_path_root}/test_metrics.csv"

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
#   grep <your-job-name> |
#   awk '{print $1}' |
#   xargs scancel


def parser_args(parser):
    parser.add_argument(
        "-target",
        type=str,
        default="ABCD",
        metavar="D",
        help=(
            "Target column in y (e.g. 'AFATINIB' for depmap_gdsc, 'TAMOXIFEN' for "
            "depmap_ctrp)."
        ),
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="depmap_gdsc",
        metavar="S",
        help=(
            "Dataset name; supported by this script: 'depmap_gdsc', 'depmap_ctrp'."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        type=str,
        help=(
            "User-defined experiment tag propagated into output paths "
            "(e.g. 'h16' for 16 held-out cancer types, 'artemis_pbcp' for TransNEO external validation)."
        ),
    )
    parser.add_argument(
        "-constraints",
        default=None,
        type=str,
        nargs="+",
        help=(
            "If specified, uses encoder with these constraints rather than using "
            "defaults/selecting automatically."
        ),
    )
    parser.add_argument(
        "-walltime",
        default="06:00:00",
        type=str,
        help="Walltime per job in HH:MM:SS format.",
    )
    parser.add_argument(
        "--newstudy",
        default=False,
        action="store_true",
        help="If specified, resubmits all jobs regardless of previous completion.",
    )

    return parser


if __name__ == "__main__":
    # 1. Parse args
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    # 2. Submit jobs
    main(args)
