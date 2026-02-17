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
source /home/dk538/.bashrc
module load hdf5/1.8.2
conda activate slurm-torch-2

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
    # Don't run on GPU
    # else:
    # script_args = fr'''{script_args} --cuda'''
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
                    # job_path_val = f"{job_path_root}/z_pred_val_0_best_s{seed}.csv"

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
                    # else:
                    # print(f"Job previously completed: {job_path}")


# To cancel these jobs, run the below
# squeue --me --states=ALL --Format=jobid,name --noheader |
#   grep dk538-ssvae |
#   awk '{print $1}' |
#   xargs scancel


def parser_args(parser):
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
        "-walltime",
        default="24:00:00",
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
