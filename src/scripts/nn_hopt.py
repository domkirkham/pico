import sys

wd_path = "/home/dk538/rds/hpc-work/pico/src"
sys.path.append(wd_path)

import argparse

import torch
import sys
import numpy as np
import pandas as pd
import random

from utils.data_utils import Manual, get_data_loaders, process_data
from models.pico import BaselineNN

import os
import json
import shutil

# hyperopt imports
import optuna
from optuna.trial import TrialState


def main(trial, x, s, y, args, hopt=True):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if hopt:
        cv_num = 5
        val_split = 1 / cv_num
    else:
        # Default val split: get_data_loaders recombines data
        # Only run one fold
        cv_num = 1
        val_split = 0.2

    # Change to grid search
    args.batch_size = 2 ** (trial.suggest_int("batch_size", 3, 6, step=1))
    args.learning_rate = 10 ** (trial.suggest_float("learning_rate", -4, -2, step=1))
    args.weight_decay = 10 ** (trial.suggest_float("weight_decay", -4, -2, step=1))
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    args.model_size = trial.suggest_categorical("model_size", ["linear", "1", "2", "3"])

    trial.set_user_attr("seed", args.seed)

    # Default filtering for variance above 1 in x
    dataset_params = {"var_filt_x": 1, "var_filt_s": None}

    # MAKING DATASET
    dataset = Manual(
        x=x,
        s=s,
        y=y,
        constraints=[s.dropna(axis=1).columns[0]],
        target=args.target,
        params=dataset_params,
        verbose=True,
    )

    # IF HOPT CHECK IF TRIAL HAS BEEN PERFORMED
    if hopt:
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
                    f"{args.save_folder}/args_{previous_trial.number}.txt",
                    f"{args.save_folder}/args_{trial.number}.txt",
                )
                return previous_trial.value

    fold_min_losses = []
    fold_best_epochs = []

    for curr_fold in range(cv_num):
        if hopt:
            print(f"Fold {curr_fold}:")
        else:
            print("Retraining best model:")
        data_loaders = get_data_loaders(
            dataset,
            args.test_samples,
            args.batch_size,
            fold=curr_fold,
            seed=args.seed,
            val_split=val_split,
            stage=args.stage,
            save_folder=args.save_folder,
            hopt=hopt,
            verbose=True,
        )

        x_0, _, _, _, _, _ = dataset[0]
        input_dim = x_0.shape[0]

        model = BaselineNN(
            reg_model_size=args.model_size,
            input_dim=input_dim,
            output_dim=1,
            dropout=args.dropout,
            use_cuda=args.cuda,
        )

        # PyTorch 2 compiler
        model = torch.compile(model)

        # Only optimise parameters for regressor
        optim = torch.optim.AdamW(
            params=model.regressor.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # run inference for a certain number of epochs
        metric_dict_list = []
        # half precision training for speed
        if args.cuda:
            scaler = torch.cuda.amp.GradScaler()

        best_score = np.Inf
        counter = 0
        patience = 10
        for epoch in range(0, args.num_epochs):
            model.train()
            # compute number of batches for an epoch
            batches_per_epoch = len(data_loaders["train"])

            # initialize variables to store loss values
            epoch_losses_train = 0.0
            epoch_losses_val = 0.0

            # setup the iterators for training data loaders
            train_iter = iter(data_loaders["train"])

            # TRAINING
            for i in range(batches_per_epoch):
                (xs, ss, cs, ys, idxs, sts) = next(train_iter)

                if args.cuda:
                    xs, ss, cs, ys, idxs, sts = (
                        xs.cuda(non_blocking=True),
                        ss.cuda(non_blocking=True),
                        cs,
                        ys.cuda(non_blocking=True),
                        idxs,
                        sts,
                    )

                if len(ys.shape) == 1:
                    ys = ys.unsqueeze(dim=1)

                if args.cuda:
                    with torch.cuda.amp.autocast():
                        loss = model.regressor_loss(xs, ys)
                else:
                    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                        loss = model.regressor_loss(xs, ys)

                epoch_losses_train += loss.detach().item() / batches_per_epoch

                if args.cuda:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

                optim.zero_grad(set_to_none=True)

            # VALIDATION
            # Validation losses are calculated on the entire validation set, whereas training metrics are averaged
            # over batches
            # In the validation set a smaller final batch with high/low error can affect metrics a lot
            if hopt:
                model.eval()
                model.regressor.eval()
                with torch.no_grad():
                    val_rmse, val_pearsonr, val_spearmanr = model.calculate_metrics(
                        data_loaders["val"], 1
                    )

                    val_iter = iter(data_loaders["val"])
                    val_batches = len(data_loaders["val"])
                    for i in range(val_batches):
                        (xv, sv, cs, yv, idxv, stv) = next(val_iter)
                        if args.cuda:
                            xv, sv, cs, yv, idxv, stv = (
                                xv.cuda(non_blocking=True),
                                sv.cuda(non_blocking=True),
                                cs,
                                yv.cuda(non_blocking=True),
                                idxv,
                                stv,
                            )
                        val_loss = model.regressor_loss(xv, yv)
                        epoch_losses_val += val_loss.detach().item() / val_batches

                metric_dict_list.append(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_losses_train,
                        "val_loss": epoch_losses_val,
                        "val_rmse": val_rmse[0],
                        "val_pearsonr": val_pearsonr[0],
                        "val_spearmanr": val_spearmanr[0],
                    }
                )

                # Printing the epoch results in a neat table format
                print(f"\n[Seed {args.seed} Fold {curr_fold} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Sup Loss:':<30}{epoch_losses_train:.4f}")
                print(f"{'Val Loss:':<30}{epoch_losses_val:.4f}")
                print(f"{'Val RMSE:':<30}{val_rmse[0]:.4f}")
                print(f"{'Val Pearson r:':<30}{val_pearsonr[0]:.4f}")
                print(f"{'Val Spearman r:':<30}{val_spearmanr[0]:.4f}")
                print(f"{'-' * 50}")

                # EARLY STOPPING COUNTER -- NO MODEL SAVING FOR HOPT
                # Use loss for early stopping
                # Check if val_loss improves or not, always save first epoch in case no increase
                if val_loss < best_score:
                    # val_loss improves, we update the latest best_score
                    best_score = val_loss
                    print("Best model:")
                    torch.save(
                        {"state_dict": model.state_dict()},
                        f"{save_folder}/best_model_{curr_fold}_{trial.number}.tar",
                    )
                    # torch.save(model, f"{save_folder}/best_model_{curr_fold}_{trial.number}.pt")
                    # model.save_models(save_folder, fold=trial.number)
                    counter = 0
                else:
                    # val_loss does not improve, we increase the counter
                    # stop training if it exceeds the amount of patience
                    counter += 1
                    if counter >= patience:
                        print("Early stopping...")
                        break

            else:
                metric_dict_list.append(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_losses_train,
                    }
                )

                # Printing the epoch results in a neat table format
                print(f"\n[Seed {args.seed} Fold {curr_fold} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Sup Loss:':<30}{epoch_losses_train:.4f}")
                print(f"{'-' * 50}")

        if hopt:
            # SAVE FOLD LOSSES
            losses_df = pd.DataFrame.from_dict(metric_dict_list)
            losses_df.to_csv(f"{save_folder}/losses_{trial.number}_{curr_fold}.csv")

            # GET BEST EPOCH
            best_epoch = losses_df["val_loss"].idxmin() + 1

            fold_best_epochs.append(best_epoch)
            fold_min_losses.append(np.nanmin(losses_df["val_loss"].tolist()))
        else:
            # SAVE LOSSES FROM FOLD TO CSV
            losses_df = pd.DataFrame.from_dict(metric_dict_list)
            losses_df.to_csv(f"{save_folder}/losses_best_retrain.csv")

    if hopt:
        # SAVE TRIAL ARGS -- INCLUDING BEST VAL LOSS AND BEST EPOCH
        args.val_loss = float(np.nanmean(fold_min_losses))
        args.best_epoch = round(np.mean(fold_best_epochs))
        with open(f"{save_folder}/args_{trial.number}.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        # SAVE TRIAL RESULTS
        folds_df = pd.DataFrame(
            {
                "fold": range(cv_num),
                "val_rmse": fold_min_losses,
                "best_epoch": fold_best_epochs,
            }
        )
        folds_df.to_csv(f"{save_folder}/cv_results_{trial.number}.csv")

        # SAVE MEAN OF TRIAL BEST EPOCHS FOR RETRAINING
        trial.set_user_attr("num_epochs", np.mean(fold_best_epochs))

        # PRINT TRIAL RESULTS
        print(
            f"Trial {trial.number} val loss ({cv_num}-fold CV): {np.nanmean(fold_min_losses)} ({np.nanstd(fold_min_losses)})"
        )

        # RETURN MEAN LOSS ACROSS CVS FOR OPTUNA
        return np.nanmean(fold_min_losses)

    else:
        # SAVE MODEL AFTER FINAL EPOCH
        torch.save(
            {"state_dict": model.state_dict()},
            f"{save_folder}/best_model_{args.seed}.tar",
        )
        model.save_models(save_folder, seed=args.seed)

        # GENERATE PREDICTIONS FOR TEST SET
        with torch.no_grad():
            if "test" in data_loaders.keys():
                model.eval()
                test_rmse, test_pearson, test_spearman = model.calculate_metrics(
                    data_loaders["test"], 1
                )
                print(f"\n[Seed {args.seed}] Test")
                print(f"{'-' * 30}")
                print(f"{'Metric':<20}{'Value'}")
                print(f"{'-' * 30}")
                print(f"{'RMSE':<20}{test_rmse[0]:.4f}")
                print(f"{'Pearson r':<20}{test_pearson[0]:.4f}")
                print(f"{'Spearman r':<20}{test_spearman[0]:.4f}")
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

        # SAVE BEST ARGS
        with open(f"{save_folder}/args_best.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

        return {
            "test_rmse": test_rmse[0],
            "test_pearson": test_pearson[0],
            "test_spearman": test_spearman[0],
        }


def parser_args(parser):
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="use GPU(s) to speed up training",
    )
    parser.add_argument(
        "-n", "--num-epochs", default=200, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-target",
        type=str,
        default="ABCD",
        metavar="T",
        help="Target column from y",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="depmap_gdsc",
        metavar="S",
        help="Dataset name (e.g. depmap_gdsc)",
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
        "--newstudy",
        default=False,
        action="store_true",
        help="Whether to always start a new study in optuna",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    # Create folder to save
    save_folder = (
        f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/nn"
    )
    # save_folder = f"./data/outputs/{timestr}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # TASK SPECIFIC SECTION
    x, s, c, y, test_samples = process_data(
        dataset=args.dataset, wd_path=wd_path, experiment=args.experiment
    )
    args.test_samples = test_samples
    # END OF TASK SPECIFIC SECTION

    # Set stage for data loaders
    args.stage = "p"
    args.save_folder = save_folder
    # CREATE OPTUNA STUDY
    # If specified to start new study and the study already exists, delete the study
    if args.newstudy:
        try:
            optuna.delete_study(
                storage=f"sqlite:////{wd_path}/data/outputs/{args.dataset}/icovae_optuna.db",
                study_name="_".join(save_folder.split("/")[-3:]) + f"_s{args.seed}",
            )
        except UserWarning("No study found to delete..."):
            pass

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:////{wd_path}/data/outputs/{args.dataset}/nn_optuna.db",
        engine_kwargs={"connect_args": {"timeout": 100}},
    )
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50, multivariate=True, seed=args.seed
        ),
        storage=storage,
        study_name="_".join(save_folder.split("/")[-3:]) + f"_s{args.seed}",
        load_if_exists=True,
    )

    n_complete_trials = len(study.trials)

    def func(trial):
        """
        Function to be optimized by optuna
        :param trial: optuna trial object
        :return: validation loss for the trial
        """
        # Call main function with trial and args and global variables
        return main(trial, x=x, s=s, y=y, args=args, hopt=True)

    study.optimize(func, n_trials=100 - n_complete_trials)

    study_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).sort_values(by="value")
    study_df.to_csv(f"{save_folder}/opt_study_results_s{args.seed}.csv")

    # REFIT MODEL USING BEST TRIAL
    args.num_epochs = int(np.round(study.best_trial.user_attrs["num_epochs"]))
    metrics = main(study.best_trial, x=x, s=s, y=y, args=args, hopt=False)
    # metrics = {key: [val] for key, val in metrics.items()}

    # REFIT FOR 10 RANDOM SEEDS
    metrics = {"test_rmse": [], "test_pearson": [], "test_spearman": []}
    for seed in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        args.seed = seed
        print(f"\n[Seed {args.seed}]")
        curr_metrics = main(study.best_trial, x=x, s=s, y=y, args=args, hopt=False)
        for key, val in metrics.items():
            metrics[key].append(curr_metrics[key])

    # SAVE METRICS TO CSV
    metrics_df = pd.DataFrame.from_dict(metrics).to_csv(
        f"{save_folder}/test_metrics.csv"
    )

    # KEEP RECORD OF REFITTING
    study.best_trial.set_user_attr("retrained", True)
