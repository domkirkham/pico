import sys

wd_path = "/home/dk538/rds/hpc-work/pico/src"
sys.path.append(wd_path)

import argparse
import torch

import numpy as np
import pandas as pd

from utils.data_utils import Manual, get_data_loaders, process_data
from models.pico import PiCoSK, PiCoCox

import os
import json
import random
import shutil

# hyperopt imports
import optuna
from optuna.trial import TrialState


def main(trial, x, s, c, y, args, hopt=True, save_preds=False):
    """
    Train PiCo
    :param args: arguments for PiCo
    :return: None
    """

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = 32

    if hopt:
        cv_num = 5
        val_split = 1 / cv_num
    else:
        # Default val split: get_data_loaders recombines data
        # Only run one fold
        cv_num = 1
        val_split = 0.2

    # SELECT PARAMETERS
    if args.reg == "ElasticNet":
        args.alpha = trial.suggest_categorical(
            "alpha",
            np.logspace(-4, 4, 40),
        )
        args.l1_ratio = trial.suggest_categorical(
            "l1_ratio",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
    elif args.reg == "CoxPH":
        args.alpha = trial.suggest_categorical(
            "alpha",
            np.logspace(-4, 4, 40),
        )
        args.l1_ratio = trial.suggest_categorical(
            "l1_ratio",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
    elif args.reg == "LogisticRegression":
        args.alpha = trial.suggest_categorical(
            "alpha",
            np.logspace(-4, 4, 40),
        )
        args.l1_ratio = trial.suggest_categorical(
            "l1_ratio",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
    elif args.reg == "SVR":
        args.c = trial.suggest_categorical("c", np.logspace(-3, 3, 30))
        args.eps = trial.suggest_categorical("eps", np.logspace(-3, 0, 15))
        args.kernel = "linear"
    elif args.reg == "SVC":
        args.c = trial.suggest_categorical("c", np.logspace(-3, 3, 30))
        args.eps = trial.suggest_categorical("eps", np.logspace(-3, 0, 15))
        args.kernel = "linear"
    else:
        raise ValueError(
            "Unrecognised model type. Please select from: ElasticNet, SVR, LogisticRegression, SVC"
        )

    trial.set_user_attr("enc", args.enc)
    trial.set_user_attr("seed", args.seed)

    # Default filtering for variance above 1 in x
    dataset_params = {"var_filt_x": 1.0, "var_filt_s": None}

    # MAKING DATASET
    dataset = Manual(
        x=x,
        s=s,
        y=y,
        c=c,
        constraints=args.constraints,
        target=args.target,
        confounders=args.confounders,
        params=dataset_params,
        duration_event=args.duration_event,
        verbose=True,
    )

    # IF HOPT CHECK IF TRIAL HAS BEEN PERFORMED
    if hopt and not save_preds:
        for previous_trial in trial.study.trials:
            if (
                previous_trial.state == TrialState.COMPLETE
                and trial.params == previous_trial.params
            ):
                print(
                    f"Duplicated trial: {trial.params}, return {previous_trial.value}"
                )
                # load args for previous trial and then save them for this trial
                shutil.copyfile(
                    f"{args.save_folder}/args_{previous_trial.number}_s{args.seed}.txt",
                    f"{args.save_folder}/args_{trial.number}_s{args.seed}.txt",
                )
                return previous_trial.value

    if args.cuda:
        map_loc = torch.device("cuda:0")
    else:
        map_loc = "cpu"

    # LOAD ENCODER USE IN ALL FOLDS
    encoder = torch.load(
        f"{args.enc_path}/encoder_{args.seed}.pt", map_location=map_loc
    )

    for curr_fold in range(cv_num):
        if not hopt:
            print(f"\n[Seed {args.seed}] Retraining best model...")
        data_loaders = get_data_loaders(
            dataset,
            args.test_samples,
            args.batch_size,
            fold=curr_fold,
            seed=args.seed,
            val_split=val_split,
            stage=args.stage,
            hopt=hopt,
            verbose=(curr_fold == 0),
            num_workers=1,
            pin_memory=False,
        )

        x, _, _, _, _, _ = dataset[0]
        # input_dim = x.shape[0]
        # print(f"Input dim: {input_dim}")

        # CREATE PICO MODEL
        if args.reg == "ElasticNet":
            model = PiCoSK(
                encoder=encoder,
                model=args.reg,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
                use_cuda=args.cuda,
                norep=args.norep,
            )
        elif args.reg == "SVR":
            model = PiCoSK(
                encoder=encoder,
                model=args.reg,
                c=args.c,
                eps=args.eps,
                use_cuda=args.cuda,
                kernel=args.kernel,
            )
        elif args.reg == "LinearRegression":
            model = PiCoSK(
                encoder=encoder, model=args.reg, use_cuda=args.cuda, norep=args.norep
            )
        elif args.reg == "SVC":
            model = PiCoSK(
                encoder=encoder,
                model=args.reg,
                use_cuda=args.cuda,
                c=args.c,
                eps=args.eps,
                kernel=args.kernel,
            )
        elif args.reg == "LogisticRegression":
            model = PiCoSK(
                encoder=encoder,
                model=args.reg,
                use_cuda=args.cuda,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
                norep=args.norep,
            )
        elif args.reg == "CoxPH":
            if args.strata is not None:
                strata = [f"c_{args.confounders.index(st)}" for st in args.strata]
            else:
                strata = None
            model = PiCoCox(
                encoder=encoder,
                model=args.reg,
                use_cuda=args.cuda,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
                norep=args.norep,
                duration_col=args.duration_event[0],
                event_col=args.duration_event[1],
                strata=strata,
            )

        # PyTorch 2 compiler for encoder for inference speed
        model = torch.compile(model)

        model.eval()

        metric_names = {
            "classification": ["cross_entropy", "f1", "aupr", "auroc"],
            "regression": ["rmse", "pearsonr", "spearmanr"],
            "survival": ["neg_log_likelihood", "concordance_index"],
        }

        with torch.no_grad():
            # FIT REGRESSOR
            model.fit_regressor(data_loaders["train"])

            if hopt:
                # GENERATE PREDICTIONS FOR VAL SET
                metrics = model.calculate_metrics(data_loaders["val"], 1)
                if save_preds:
                    model.generate_predictions(
                        data_loaders["val"],
                        args.save_folder,
                        suffix=f"val_{curr_fold}_best_s{args.seed}",
                    )

                print(f"\n[Seed {args.seed} Fold {curr_fold}] Validation")
                print(f"{'-' * 40}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 40}")
                for i in range(len(metrics)):
                    print(
                        f"{metric_names[model.metric_type][i]:<30}{metrics[i][0]:.4f}"
                    )
                print(f"{'-' * 40}")

                if curr_fold == 0:
                    fold_metrics = {
                        metric_name: [metrics[i][0]]
                        for i, metric_name in enumerate(metric_names[model.metric_type])
                    }
                else:
                    for i, metric_name in enumerate(metric_names[model.metric_type]):
                        fold_metrics[metric_name].append(metrics[i][0])

            else:
                # GENERATE PREDICTIONS FOR TEST SET
                with torch.no_grad():
                    if "test" in data_loaders.keys():
                        model.eval()
                        metrics = model.calculate_metrics(data_loaders["test"], 1)
                        print(f"\n[Seed {args.seed}] Test")
                        print(f"{'-' * 30}")
                        print(f"{'Metric':<20}{'Value'}")
                        print(f"{'-' * 30}")
                        for i in range(len(metrics)):
                            print(
                                f"{metric_names[model.metric_type][i]:<30}{metrics[i][0]:.4f}"
                            )
                        print(f"{'-' * 30}")

                with torch.no_grad():
                    if "test" in data_loaders.keys():
                        model.generate_predictions(
                            data_loaders["test"],
                            save_folder,
                            suffix=f"test_s{args.seed}",
                        )
                    model.generate_predictions(
                        data_loaders["train"], save_folder, suffix=f"train_s{args.seed}"
                    )

    if hopt:
        # SAVE TRIAL RESULTS
        folds_df_dict = {"fold": range(cv_num)}
        for key, val in fold_metrics.items():
            folds_df_dict[f"val_{key}"] = val

        if save_preds:
            folds_df = pd.DataFrame.from_dict(folds_df_dict)

            # Don't need to save the below
            folds_df.to_csv(f"{args.save_folder}/cv_results_best_s{args.seed}.csv")

        # SAVE TRIAL ARGS -- INCLUDING BEST VAL LOSS AND BEST EPOCH
        # val loss is first listed metric in metric_names
        args.val_loss = float(
            np.nanmean(folds_df_dict[f"val_{metric_names[model.metric_type][0]}"])
        )
        # with open(f"{args.save_folder}/args_{trial.number}_s{args.seed}.txt", "w") as f:
        #     json.dump(args.__dict__, f, indent=2)

        # RETURN FOR OPTUNA
        return args.val_loss

    else:
        # SAVE MODEL COEFFS
        model.save_models(seed=args.seed, path=save_folder)

        # SAVE ARGS
        with open(f"{save_folder}/args_best_s{args.seed}.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        test_metrics = {
            f"test_{metric_name}": metrics[i][0]
            for i, metric_name in enumerate(metric_names[model.metric_type])
        }

        return test_metrics


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
        "-enc",
        type=str,
        default="iCoVAE",
        metavar="M",
        help="Encoder type",
    )
    parser.add_argument(
        "-reg",
        type=str,
        default="ElasticNet",
        metavar="M",
        help="Regressor type",
    )
    parser.add_argument(
        "-constraints",
        default=None,
        type=str,
        nargs="+",
        help="If specified, uses encoder with these constraints rather than using defaults/selecting automatically",
    )
    parser.add_argument(
        "--confounders",
        default=None,
        type=str,
        nargs="+",
        help="Confounders for prediction model",
    )
    parser.add_argument(
        "--duration-event",
        default=None,
        type=str,
        nargs=2,
        help="Duration and event column suffix i.e. target_duration and target_event",
    )
    parser.add_argument(
        "--strata",
        default=None,
        type=str,
        nargs="+",
        help="Strata if using CoxPH. Should also be a confounder.",
    )
    parser.add_argument("-seed", default=4563, type=int, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data path")
    parser.add_argument(
        "--norm",
        default=False,
        action="store_true",
        help="Minmax normalize input data in range 0-1",
    )
    parser.add_argument(
        "-ld",
        "--lindec",
        default=False,
        action="store_true",
        help="Use a linear decoder?",
    )
    parser.add_argument(
        "--filt",
        type=str,
        default="uni_var",
        help="Filter type to use on input expression data",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        type=str,
        help="Experiment to run (user defined in data loading class)",
    )
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="use GPU(s) for encoder inference (not suggested)",
    )
    parser.add_argument(
        "--norep",
        default=False,
        action="store_true",
        help="Whether to make a prediction model without using a representation",
    )
    parser.add_argument(
        "--newstudy",
        default=False,
        action="store_true",
        help="Whether to always start a new study in optuna",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="Test by running a single trial with fixed hyperparameters",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    # Unless specified, assume model type is same drug with same seed
    # Otherwise leave as is
    if args.enc == "VAE":
        args.enc_path = f"{wd_path}/data/outputs/{args.dataset}/{args.target}"
        if args.experiment is not None:
            args.enc_path = f"{args.enc_path}/{args.experiment}/vae"
        else:
            args.enc_path = f"{args.enc_path}/default/vae"
    elif args.enc == "iCoVAE":
        args.enc_path = f"{wd_path}/data/outputs/{args.dataset}/{args.target}"
        if args.experiment is not None:
            args.enc_path = f"{args.enc_path}/{args.experiment}/icovae"
        else:
            args.enc_path = f"{args.enc_path}/default/icovae"

    if (args.constraints is not None) and (args.enc == "iCoVAE"):
        # If self defined constraints suffix with first constraint then number of constraints
        args.enc_path = f"{args.enc_path}_{args.constraints[0]}_{len(args.constraints)}"

    if args.lindec:
        # If using a linear decoder
        args.enc_path = f"{args.enc_path}_ld"

    save_ext = args.enc_path.split("/")[-1]

    if args.confounders is not None:
        save_ext = f"{save_ext}_{args.confounders[0]}_{len(args.confounders)}"

    if args.norep:
        save_ext = f"{save_ext}_norep"

    # LOAD ARGUMENTS FOR ENCODER
    with open(f"{args.enc_path}/args_best.txt", "r") as f:
        enc_args = json.load(f)

    # These are the same for every seed
    if args.enc == "VAE":
        args.constraints = enc_args["curr_constraints"]
    elif args.enc == "iCoVAE":
        args.constraints = enc_args["curr_constraints"]
    args.test_samples = enc_args["test_samples"]

    # enc_args = pd.Series(enc_args).to_frame().T
    # Get constraints from enc args

    # LOAD BEST TRIAL FROM OPTUNA DB
    # We don't need to know anything about the encoder arguments in this case

    if args.reg == "ElasticNet":
        search_space = {
            "alpha": np.logspace(-4, 4, 40),
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        n_trials = 400

    elif args.reg == "CoxPH":
        search_space = {
            "alpha": np.logspace(-4, 4, 40),
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        n_trials = 400
    elif args.reg == "LogisticRegression":
        search_space = {
            "alpha": np.logspace(-4, 4, 40),
            "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        n_trials = 400
    elif args.reg == "SVR":
        search_space = {
            "c": np.logspace(-3, 3, 30),
            "eps": np.logspace(-3, 0, 15),
        }
        n_trials = 450
    elif args.reg == "SVC":
        search_space = {
            "c": np.logspace(-3, 3, 30),
            "eps": np.logspace(-3, 0, 15),
        }
        n_trials = 450

    if args.test:
        search_space = {"alpha": [-4], "l1_ratio": [1.0]}
        n_trials = 1

    # timestr = time.strftime("%Y%m%d_%H%M%S")
    save_folder = f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/pico/{args.reg}_{save_ext}"
    # save_folder = f"./data/outputs/{timestr}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # TASK SPECIFIC SECTION
    x, s, c, y, test_samples = process_data(
        dataset=args.dataset, wd_path=wd_path, experiment=args.experiment
    )
    # Check dataset is being processed in the same way as in iCoVAE step
    # assert sorted(test_samples) == sorted(args.test_samples)
    # Fix this
    args.test_samples = test_samples
    # END OF TASK SPECIFIC SECTION

    # SET OTHER ARGUMENTS
    # iCoVAE stage
    args.stage = "p"
    args.save_folder = save_folder

    # If specified to start new study and the study already exists, delete the study
    if args.newstudy:
        try:
            optuna.delete_study(
                storage=f"sqlite:////{save_folder}/pico_optuna.db",
                study_name="_".join(save_folder.split("/")[-5:]) + f"_s{args.seed}",
            )
        except UserWarning("No study found to delete..."):
            pass

    # These models fit quickly and search space is small so we fit all combinations of hyperparams
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:////{save_folder}/pico_optuna.db",
        engine_kwargs={"connect_args": {"timeout": 100}},
    )
    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
        storage=storage,
        study_name="_".join(save_folder.split("/")[-5:]) + f"_s{args.seed}",
        load_if_exists=True,
    )

    def func(trial):
        """
        Function to be optimized by optuna
        :param trial: optuna trial object
        :return: validation loss for the trial
        """
        # Call main function with trial and args and global variables
        return main(trial, x=x, s=s, c=c, y=y, args=args)

    n_complete_trials = len(study.trials)

    study.optimize(func, n_trials=n_trials - n_complete_trials)

    study_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).sort_values(by="value")
    study_df.to_csv(f"{args.save_folder}/opt_study_results_s{args.seed}.csv")

    # REFIT MODEL USING BEST TRIAL
    # RUN HOPT AGAIN BUT SAVE PREDICTIONS
    main(study.best_trial, x=x, s=s, c=c, y=y, args=args, hopt=True, save_preds=True)
    # REFIT ON ALL SAMPLES
    metrics = main(study.best_trial, x=x, s=s, c=c, y=y, args=args, hopt=False)
    metrics = {key: [val] for key, val in metrics.items()}

    # REFIT FOR 10 RANDOM SEEDS
    # for seed in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    #     args.seed = seed
    #     print(f"\n[Seed {args.seed}]")
    #     curr_metrics = main(study.best_trial, x=x, s=s, y=y, args=args, hopt=False)
    #     if seed == 10:
    #         metrics = curr_metrics.copy()
    #         # Make values into a list
    #         metrics = {key: [val] for key, val in metrics.items()}
    #     else:
    #         for key, val in metrics.items():
    #             metrics[key].append(curr_metrics[key])

    # SAVE METRICS TO CSV
    metrics_df = pd.DataFrame.from_dict(metrics).to_csv(
        f"{save_folder}/test_metrics_s{args.seed}.csv"
    )

    # KEEP RECORD OF REFITTING
    study.best_trial.set_user_attr("retrained", True)
