"""Generate and submit SLURM jobs for the no-encoder scikit-learn baseline on DepMap drugs.

This script writes a SLURM submission script for each combination of regressor
('ElasticNet', 'RandomForestRegressor', 'SVR'), PCA on/off, drug target (sweeps
the full hard-coded `target_dict` of GDSC drugs and their target classes used
in the H16 holdout experiments) and seed (4563), then submits each job via
`sbatch`. Submitted jobs invoke src/scripts/sk_hopt.py with the matching
command-line arguments.

Inputs are command-line arguments listing the dataset, optional experiment tag,
walltime and a --newstudy resubmission flag. Outputs are SLURM scripts written
to <wd_path>/submissions/ and the queued jobs themselves; the existing
test_metrics.csv check is currently disabled (the `and False` guard) so all
combinations are always submitted.

Example:
    python src/scripts/schedule_jobs_depmap_sk.py -dataset depmap_gdsc \\
        --experiment h16 -walltime 24:00:00
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
    dataset, reg, target, seed, experiment=None, pca=False, walltime="02:00:00"
):
    submission_preamble = rf"""#!/bin/bash
#SBATCH -J JOB-NAME-HERE
#SBATCH -A YOUR-ACCOUNT-HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={walltime}
#SBATCH --output=/home/USER/pico/slurm_out/sk_hopt/%j.out
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

application="python -u ./scripts/sk_hopt.py"

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

    script_args = rf"""CMD="$application -reg {reg} -target {target} -dataset {dataset} -seed {seed}"""
    if experiment is not None:
        script_args = rf"""{script_args} --experiment {experiment}"""
    if pca:
        script_args = rf"""{script_args} --pca"""
    script_args = rf'''{script_args}"'''

    submission = rf"""{submission_preamble}
{script_args}
{submission_post}"""

    return submission


def main(args):
    regs = ["ElasticNet", "RandomForestRegressor", "SVR"]
    pca_opt = [False, True]
    seeds = [4563]

    target_dict = {
        "AZD5991": "MCL1/BCL2",
        "Alpelisib": "PI3K",
        "AZD8186": "PI3K",
        "Gefitinib": "EGFR/ERBB2",
        "Lapatinib": "EGFR/ERBB2",
        "Sorafenib": "VEGFR",
        "Docetaxel": "Taxane",
        "Paclitaxel": "Taxane",
        "Taselisib": "PI3K",
        "AZD6482": "PI3K",
        "Palbociclib": "CDK4/6",
        "AZD3759": "EGFR/ERBB2",
        "Afatinib": "EGFR/ERBB2",
        "Afuresertib": "AKT",
        "Serdemetan": "MDM2",
        "Oxaliplatin": "Platinum",
        "GSK1904529A": "IGF1R",
        "Buparlisib": "PI3K",
        "Linsitinib": "IGF1R",
        "Ipatasertib": "AKT",
        "CZC24832": "PI3K",
        "Sabutoclax": "MCL1/BCL2",
        "MK-8776": "CHEK1",
        "Ribociclib": "CDK4/6",
        "Cisplatin": "Platinum",
        "Osimertinib": "EGFR/ERBB2",
        "Erlotinib": "EGFR/ERBB2",
        "AZD6738": "ATR",
        "Olaparib": "PARP",
        "Niraparib": "PARP",
        "Veliparib": "PARP",
        "MK-1775": "WEE1",
        "Cyclophosphamide": "Alkylating agent",
        "5-Fluorouracil": "Antimetabolite",
        "Epirubicin": "Anthracycline",
        "Tamoxifen": "ER",
        "Methotrexate": "Antimetabolite",
        "Venetoclax": "MCL1/BCL2",
        "JQ1": "BRD4",
        "PD173074": "FGFR",
        "Sapitinib": "EGFR/ERBB2",
        "AZD4547": "FGFR",
        "Vorinostat": "HDAC",
        "Refametinib": "MEK",
        "Selumetinib": "MEK",
        "Trametinib": "MEK",
        "Axitinib": "VEGFR",
        "GSK2830371A": "PPM1D",
        # "CCT007093": "PPM1D",
        "Gemcitabine": "Antimetabolite",
        "Irinotecan": "TOP1",
        "VE-822": "ATR",
        "5-Fluorouracil": "Antimetabolite",
        "Crizotinib": "ALK/ROS1",
        "Cytarabine": "Antimetabolite",
        "Entinostat": "HDAC",
        "Foretinib": "VEGFR",
        "Fulvestrant": "ER",
        "Motesanib": "VEGFR",
        "Navitoclax": "MCL1/BCL2",
        "PD173074": "FGFR",
        "Pyridostatin": "G4",
        "Temsirolimus": "mTOR",
        "Tanespimycin": "HSP90",
        "Uprosertib": "AKT",
        "AZD5363": "AKT",
        "Dabrafenib": "BRAF",
        "Temozolomide": "Alkylating agent",
        "Vinblastine": "Vinca alkyloid",
        "Vinorelbine": "Vinca alkyloid",
    }
    # FULL DRUG LIST FOR H16
    targets = sorted(target_dict.keys())
    print(targets)

    for target in targets:
        target = target.upper()
        for seed in seeds:
            for reg in regs:
                for pca in pca_opt:
                    job_path_root = f"{wd_path}/data/outputs/{args.dataset}/{target}/{args.experiment}/{reg}"
                    if pca:
                        job_path_root = f"{job_path_root}_pca"

                    job_path = f"{job_path_root}/test_metrics.csv"

                    if os.path.exists(job_path) and not args.newstudy and False:
                        print(f"Job previously completed: {job_path}")
                    else:
                        if args.dataset == "depmap_gdsc":
                            script = submission_pico_sk_gdsc(
                                dataset=args.dataset,
                                reg=reg,
                                target=target,
                                seed=seed,
                                experiment=args.experiment,
                                pca=pca,
                                walltime=args.walltime,
                            )
                        if args.dataset == "depmap_ctrp":
                            script = submission_pico_sk_gdsc(
                                dataset=args.dataset,
                                reg=reg,
                                target=target,
                                seed=seed,
                                experiment=args.experiment,
                                pca=pca,
                                walltime=args.walltime,
                            )

                        script_name = f"{wd_path}/submissions/submit_{args.dataset}_{reg}_{pca}_{target}_{args.experiment}_{seed}"
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
        "-walltime",
        default="24:00:00",
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
