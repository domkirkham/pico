import sys

wd_path = "/home/dk538/rds/hpc-work/pico/src"
sys.path.append(wd_path)

import random
import argparse

import torch

from utils.data_utils import Manual, get_data_loaders, get_constraints, process_data
from models.pico import iCoVAE

import numpy as np
import os
import json
import pandas as pd
import shutil

import optuna
from optuna.trial import TrialState


def main(trial, x, s, c, y, args, save_folder, hopt=True):
    """
    run inference for iCoVAE
    :param args: arguments for iCoVAE
    :param save_folder: path to save models
    :param hopt: True if doing hyperparamater optimisation, False if refitting
    :return: None
    """
    # SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = 2 ** (trial.suggest_int("batch_size", 3, 6, step=1))
    args.learning_rate = 10 ** (trial.suggest_float("learning_rate", -4, -2, step=1))
    args.weight_decay = 10 ** (trial.suggest_float("weight_decay", -4, -2, step=1))
    args.model_size = trial.suggest_categorical("model_size", ["1", "2", "3"])
    args.z_dim = 2 ** (trial.suggest_int("z_dim", 4, 6, step=1))
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    # SETTING OTHER TRIAL ATTRIBUTES
    trial.set_user_attr("model_type", "iCoVAE")
    trial.set_user_attr("seed", args.seed)
    trial.set_user_attr("constraints", args.constraints)

    # If z dim is not large enough for all constraints then fill half the latent space
    if len(args.constraints) > (args.z_dim // 2):
        args.curr_constraints = args.constraints[: (args.z_dim // 2)]
    else:
        args.curr_constraints = args.constraints

    # Save constraints used for this trial
    trial.set_user_attr("constraints", args.curr_constraints)
    trial.set_user_attr("confounders", args.confounders)

    # Default filtering for variance above 1 in x
    dataset_params = {"var_filt_x": 1.0, "var_filt_s": None}

    # MAKING DATASET
    dataset = Manual(
        x=x,
        s=s,
        c=c,
        y=y,
        constraints=args.curr_constraints,
        target=args.target,
        confounders=args.confounders,
        params=dataset_params,
    )

    # CHECK IF TRIAL HAS BEEN PERFORMED
    if hopt:
        for previous_trial in trial.study.trials:
            if (
                previous_trial.state == TrialState.COMPLETE
                and trial.params == previous_trial.params
            ):
                print(
                    f"[INFO] Duplicated trial: {trial.params}, return {previous_trial.value}"
                )
                # load args for previous trial and then save them for this trial
                shutil.copyfile(
                    f"{save_folder}/args_{previous_trial.number}.txt",
                    f"{save_folder}/args_{trial.number}.txt",
                )
                return previous_trial.value

    # In the case where we specify genes, we need to make sure we have enough zdims
    # if len(args.constraints) > args.z_dim:
    #    raise ValueError("Too many constraints specified for latent dimension. Increase z_dim or reduce number of constraints.")

    if hopt:
        cv_num = 5
        val_split = 1 / cv_num
    else:
        # Default val split: get_data_loaders recombines data
        # Only run one fold
        cv_num = 1
        val_split = 0.2

    fold_min_losses = []
    fold_best_epochs = []

    for curr_fold in range(cv_num):
        if hopt:
            print(f"[INFO] Starting Fold {curr_fold}...")
        else:
            print("[INFO] Retraining best model...")
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
        # print(f"input dim: {input_dim}")

        # .dataset.dataset accesses a Subset, then the original Dataset used to form the subset. This has the prior_fn_loc attribute
        icovae = iCoVAE(
            model_size=args.model_size,
            z_dim=args.z_dim,
            constraints=args.curr_constraints,
            input_dim=input_dim,
            dropout=args.dropout,
            s_prior_fn_loc=data_loaders["train_x_only"].dataset.dataset.s_prior_fn_loc,
            s_prior_fn_scale=data_loaders[
                "train_x_only"
            ].dataset.dataset.s_prior_fn_scale,
            use_cuda=args.cuda,
        )

        # PyTorch 2 compiler
        icovae = torch.compile(icovae)

        optim = torch.optim.AdamW(
            params=icovae.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # run inference for a certain number of epochs
        metric_dict_list = []
        # half precision training for speed
        if args.cuda:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        best_score = None
        best_epoch = 0
        counter = 0
        patience = 10
        for epoch in range(0, args.num_epochs):
            # TRAINING
            icovae.train()
            # compute number of batches for an epoch
            x_s_batches = len(data_loaders["train_x_s"])
            x_only_batches = len(data_loaders["train_x_only"])
            batches_per_epoch = x_s_batches + x_only_batches
            period_x_s_batches = int(batches_per_epoch / x_s_batches)

            # initialize variables to store loss values
            epoch_losses_x_s = 0.0
            epoch_losses_x_only = 0.0
            epoch_losses_val = 0.0
            epoch_losses_val_recon = 0.0

            # setup the iterators for training data loaders
            x_s_iter = iter(data_loaders["train_x_s"])
            x_only_iter = iter(data_loaders["train_x_only"])

            # count the number of supervised batches seen in this epoch
            ctr_x_s = 0

            if hopt:
                val_batches = len(data_loaders["val"])
            else:
                val_batches = 0

            for i in range(batches_per_epoch):
                try:
                    # whether this batch is supervised or not
                    is_supervised = (
                        i % period_x_s_batches == 0
                    ) and ctr_x_s < x_s_batches
                    # extract the corresponding batch
                    if is_supervised:
                        (xs, ss, cs, ys, idxs, sts) = next(x_s_iter)
                        ctr_x_s += 1
                    else:
                        (xs, ss, cs, ys, idxs, sts) = next(x_only_iter)

                    if args.cuda:
                        xs, ss, cs, ys, idxs, sts = (
                            xs.cuda(non_blocking=True),
                            ss.cuda(non_blocking=True),
                            cs,
                            ys.cuda(non_blocking=True),
                            idxs,
                            sts,
                        )

                    if is_supervised:
                        if args.cuda:
                            with torch.cuda.amp.autocast():
                                loss = icovae.sup(xs, ss)
                            epoch_losses_x_s += loss.detach().item()
                        else:
                            with torch.amp.autocast(
                                device_type="cpu", dtype=torch.bfloat16
                            ):
                                loss = icovae.sup(xs, ss)
                            epoch_losses_x_s += loss.detach().item()
                    else:
                        if args.cuda:
                            with torch.cuda.amp.autocast():
                                loss = icovae.unsup(xs)
                            epoch_losses_x_only += loss.detach().item()
                        else:
                            with torch.amp.autocast(
                                device_type="cpu", dtype=torch.bfloat16
                            ):
                                loss = icovae.unsup(xs)
                            epoch_losses_x_only += loss.detach().item()

                    # Only need to scale when using cuda -- CPU AMP uses bfloat16 which doesn't require scaling
                    if args.cuda:
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        optim.step()

                    optim.zero_grad(set_to_none=True)
                # Prune trial if optimisation fails
                except ValueError:
                    raise optuna.TrialPruned()

            # VALIDATION
            if hopt:
                with torch.no_grad():
                    icovae.eval()
                    val_rmse, val_pearsonr, val_spearmanr, val_kl = icovae.rmse(
                        data_loaders["val"],
                        len(args.curr_constraints),
                        map_est=True,
                        k=1,
                    )

                    val_iter = iter(data_loaders["val"])
                    for i in range(len(data_loaders["val"])):
                        (xv, sv, cv, yv, idxv, stv) = next(val_iter)
                        if args.cuda:
                            xv, sv, cv, yv, idxv, stv = (
                                xv.cuda(non_blocking=True),
                                sv.cuda(non_blocking=True),
                                cv,
                                yv.cuda(non_blocking=True),
                                idxv,
                                stv,
                            )

                        val_recon = icovae.recon_error(xv)
                        val_loss = icovae.sup(xv, sv)
                        epoch_losses_val += val_loss.detach().item()
                        epoch_losses_val_recon += val_recon.detach().item()
                    epoch_losses_val /= val_batches
                    epoch_losses_val_recon /= val_batches

                metric_dict_list.append(
                    {
                        "epoch": epoch,
                        "x_s_loss": epoch_losses_x_s / x_s_batches,
                        "x_only_loss": epoch_losses_x_only / x_only_batches,
                        "val_loss": epoch_losses_val,
                        "val_recon_loss": epoch_losses_val_recon,
                        "val_rmse": val_rmse,
                        "val_pearsonr": val_pearsonr,
                        "val_spearmanr": val_spearmanr,
                    }
                )

                # EARLY STOPPING COUNTER
                if best_score is None:
                    best_score = epoch_losses_val
                else:
                    # Check if val_loss improves or not
                    if epoch_losses_val < best_score:
                        # val_loss improves, we update the latest best_score
                        # and save the current model
                        best_score = epoch_losses_val
                        print("[INFO] Saving current model...")
                        counter = 0
                    else:
                        # val_loss does not improve, we increase the counter
                        # stop training if it exceeds the amount of patience
                        counter += 1
                        if counter >= patience:
                            print("[INFO] Early stopping...")
                            break

                # Printing the epoch results in a neat table format
                print(f"\n[Seed {args.seed} Fold {curr_fold} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Sup Loss:':<30}{epoch_losses_x_s / x_s_batches:.4f}")
                print(f"{'Val Loss:':<30}{epoch_losses_val:.4f}")
                print(f"{'Val Recon Loss:':<30}{epoch_losses_val_recon:.4f}")
                print(f"{'-' * 50}")

                # Combined Pearson and Spearman Report Table
                print(f"\n{'Constraint prediction:':<30}")
                print(f"{'-' * 80}")
                print(f"{'Constraint':<25}{'Pearson r':<20}{'Spearman r':<20}{'RMSE'}")
                print(f"{'-' * 80}")
                for i, constraint in enumerate(args.curr_constraints):
                    print(
                        f"{constraint:<25}{val_pearsonr[i]:<20.4f}{val_spearmanr[i]:<20.4f}{val_rmse[i]:.4f}"
                    )
                print(f"{'-' * 80}")
            else:
                metric_dict_list.append(
                    {
                        "epoch": epoch,
                        "x_s_loss": epoch_losses_x_s / x_s_batches,
                        "x_only_loss": epoch_losses_x_only / x_only_batches,
                        "val_loss": np.nan,
                        "val_recon_loss": np.nan,
                        "val_rmses": np.nan,
                        "val_pearsonr": np.nan,
                        "val_spearmanr": np.nan,
                    }
                )
                print(f"\n[Seed {args.seed} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Sup Loss:':<30}{epoch_losses_x_s / x_s_batches:.4f}")
                print(f"{'-' * 50}")

        if hopt:
            # SAVE LOSSES FROM FOLD TO CSV
            losses_df = pd.DataFrame.from_dict(metric_dict_list)
            losses_df.to_csv(f"{save_folder}/losses_{trial.number}_{curr_fold}.csv")

            # GET BEST EPOCH
            best_epoch = losses_df["val_loss"].idxmin() + 1

            # RETURN FOR HYPEROPT
            # filter for unstable training points, e.g. train loss >2x val loss
            losses_df["unstable"] = losses_df["x_s_loss"] > 2 * losses_df["val_loss"]
            losses_df_report = losses_df[not losses_df["unstable"]]
            val_loss_report = losses_df_report["val_loss"].tolist()
            fold_min_losses.append(np.nanmin(val_loss_report))
            fold_best_epochs.append(best_epoch)

        else:
            # SAVE LOSSES FROM FOLD TO CSV
            losses_df = pd.DataFrame.from_dict(metric_dict_list)
            losses_df.to_csv(f"{save_folder}/losses_best_retrain.csv")

    if hopt:
        # SAVE TRIAL ARGS -- INCLUDING BEST VAL LOSS AND BEST EPOCH
        args.val_loss = np.nanmean(fold_min_losses)
        args.best_epoch = round(np.mean(fold_best_epochs))
        with open(f"{save_folder}/args_{trial.number}.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2, sort_keys=True)

        # SAVE TRIAL RESULTS
        folds_df = pd.DataFrame(
            {
                "fold": range(cv_num),
                "val_loss": fold_min_losses,
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

        # RETURN MEAN LOSS ACROSS CVS FOR HYPEROPT
        return np.nanmean(fold_min_losses)

    else:
        # SAVE MODEL AFTER FINAL EPOCH
        torch.save(
            {"state_dict": icovae.state_dict()},
            f"{save_folder}/best_model_{args.seed}.tar",
        )
        icovae.save_models(save_folder, seed=args.seed)

        # # GENERATE PREDICTIONS FOR TEST SET USING FINAL MODEL
        # with torch.no_grad():
        #     if "test" in data_loaders.keys():
        #         icovae.eval()
        #         test_rmse, test_pearson, test_spearman = icovae.rmse(
        #             data_loaders["test"], len(args.constraints), map_est=True, k=1
        #         )
        #         print(f"Test RMSE: {test_rmse}")
        #         print(f"Test Pearson r: {test_pearson}")
        #         print(f"Test Spearman r: {test_spearman}")

        # SAVE BEST ARGS
        with open(f"{save_folder}/args_best.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)


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
        default="TAMOXIFEN",
        metavar="D",
        help="Target column in y.",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="depmap_gdsc",
        metavar="D",
        help="Dataset name",
    )
    # Fixed seed for hopt
    parser.add_argument("-seed", default=4563, type=int, help="Random seed")
    # Fixed arguments for hyperparameter optimisation -- use defaults
    parser.add_argument(
        "-constraints",
        default=None,
        type=str,
        nargs="+",
        help="If specified, uses these constraints rather than using defaults or selecting automatically",
    )
    parser.add_argument(
        "--confounders",
        default=None,
        type=str,
        nargs="+",
        help="Confounders used in final prediction model",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        type=str,
        help="Which experiment to run (user defined in data loading)",
    )
    parser.add_argument(
        "--col-thresh",
        default=0.7,
        type=float,
        help="Threshold for constraint collinearity filtering (only for depmap_gdsc)",
    )
    parser.add_argument(
        "--newstudy",
        default=False,
        action="store_true",
        help="Whether to always start a new study in Optuna",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="Runs a single trial in Optuna",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    # timestr = time.strftime("%Y%m%d_%H%M%S")
    if args.experiment is not None:
        save_folder = f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/icovae"
    else:
        save_folder = (
            f"{wd_path}/data/outputs/{args.dataset}/{args.target}/default/icovae"
        )

    if args.constraints is not None:
        # If self defined constraints suffix with first constraint then number of constraints
        save_folder = f"{save_folder}_{args.constraints[0]}_{len(args.constraints)}"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Put save folder in args
    args.save_folder = save_folder

    # PROCESS DATASET
    x, s, c, y, test_samples = process_data(
        dataset=args.dataset, wd_path=wd_path, experiment=args.experiment
    )
    args.test_samples = test_samples

    # SELECTING CONSTRAINTS
    if args.constraints is None:
        # Loading constraints for depmap_gdsc
        args.constraints = get_constraints(
            drug=args.target,
            dataset_name=args.dataset,
            zdim=512,
            experiment=args.experiment,
            col_thresh=args.col_thresh,
            wd_path=wd_path,
        )
    else:
        pass

    # SET OTHER ARGUMENTS
    # iCoVAE stage
    args.stage = "i"

    # CREATE OPTUNA STUDY
    # If specified to start new study and the study already exists, delete the study
    if args.newstudy:
        try:
            optuna.delete_study(
                storage=f"sqlite:////{save_folder}/icovae_optuna.db",
                study_name="_".join(save_folder.split("/")[-3:]),
            )
        except UserWarning("No study found to delete..."):
            pass

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:////{save_folder}/icovae_optuna.db",
        engine_kwargs={"connect_args": {"timeout": 1000}},
    )

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50, multivariate=True, seed=args.seed
        ),
        storage=storage,
        study_name="_".join(save_folder.split("/")[-3:]),
        load_if_exists=True,
    )

    n_complete_trials = len(study.trials)

    # RECOMMENDED PARAMETERS FOR TRIAL USE
    if args.test:
        study.enqueue_trial(
            {
                "batch_size": 3,
                "learning_rate": -3,
                "weight_decay": -2,
                "model_size": "1",
                "z_dim": 5,
                "dropout": 0.2,
            }
        )
        n_trials = 1
    else:
        n_trials = 100

    # HYPERPARAMETER OPTIMIZATION
    def func(trial):
        """
        Function to be optimized by optuna
        :param trial: optuna trial object
        :return: validation loss for the trial
        """
        # Call main function with trial and args and global variables
        return main(trial, x=x, s=s, c=c, y=y, args=args, save_folder=save_folder)

    study.optimize(
        func,
        n_trials=max(n_trials - n_complete_trials, 0),
        catch=(ValueError, RuntimeError),
    )

    study_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    ).sort_values(by="value")
    print(study_df.head(10))
    study_df.to_csv(f"{save_folder}/opt_study_results.csv")

    # REFIT MODEL USING BEST TRIAL
    args.num_epochs = int(np.round(study.best_trial.user_attrs["num_epochs"]))
    main(
        study.best_trial,
        x=x,
        s=s,
        c=c,
        y=y,
        args=args,
        save_folder=save_folder,
        hopt=False,
    )

    # REFIT FOR 10 RANDOM SEEDS
    for seed in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        args.seed = seed
        main(
            study.best_trial,
            x=x,
            s=s,
            c=c,
            y=y,
            args=args,
            save_folder=save_folder,
            hopt=False,
        )

    # KEEP RECORD OF REFITTING
    study.best_trial.set_user_attr("retrained", True)
