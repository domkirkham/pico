"""Hyperparameter optimisation and refitting for the vanilla VAE encoder baseline.

This script runs a TPE-sampled Optuna study over a vanilla VAE architecture
(no constraint head) trained as an encoder baseline against iCoVAE. Each trial
trains the model with a sampled configuration using 5-fold cross-validation and
returns the mean validation reconstruction loss. After the search completes,
the best trial's mean best-epoch count is used to refit the model on the full
training split once with the primary seed and again for ten additional random
seeds (10, 20, ..., 100).

Inputs are loaded via utils.data_utils.process_data and utils.data_utils.Manual,
which return (x, s, c, y, test_samples) for the selected dataset and
experiment. Outputs are written under
data/outputs/<dataset>/<target>/<experiment>/vae/ and include per-trial loss
CSVs, cross-validation summaries, args JSONs, the Optuna SQLite database
(vae_optuna.db), the study results CSV, and checkpoints best_model_<seed>.tar
for each refit seed.

Example:
    python src/scripts/vae_hopt.py -target AFATINIB -dataset depmap_gdsc \\
        --experiment h16 --cuda
"""

import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
import torch

wd_path = os.environ.get(
    "PICO_SRC",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.append(wd_path)

from models.pico import VanillaVAE
from utils.data_utils import Manual, get_constraints, get_data_loaders, process_data


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

    args.batch_size = trial.suggest_categorical("batch_size", [16, 32])
    args.learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3])
    args.weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 0])
    args.n_layers = trial.suggest_categorical("n_layers", [1, 2])
    args.layer_width = trial.suggest_categorical("layer_width", [128, 256])
    args.z_dim = trial.suggest_categorical("z_dim", [32, 64])
    args.dropout = 0.1

    # SETTING OTHER TRIAL ATTRIBUTES
    trial.set_user_attr("model_type", "VAE")
    trial.set_user_attr("seed", args.seed)

    # Default filtering for variance above 1 in x
    dataset_params = {"var_filt_x": 1500, "var_filt_s": None}

    trial.set_user_attr("constraints", args.constraints)

    # If z dim is not large enough for all constraints then fill half the latent space
    if len(args.constraints) > (args.z_dim // 2):
        args.curr_constraints = args.constraints[: (args.z_dim // 2)]
    else:
        args.curr_constraints = args.constraints

    # Save constraints used for this trial
    trial.set_user_attr("curr_constraints", args.curr_constraints)

    # Use dummy constraints and target if not provided
    # For clinical data studies, availability of y determines which samples are used in VAE training, so target should be supplied
    if args.confounders is None:
        # If confounders is None assume not using c
        pass
    if args.curr_constraints is None:
        args.curr_constraints = s.columns.values.tolist()[:2]
    if args.target is None:
        # Takes first column in y as target
        # This might put extra values in training set compared to iCoVAE
        args.target = y.columns.values.tolist()[0]

    # MAKING DATASET
    dataset = Manual(
        x=x,
        s=s,
        c=c,
        y=y,
        constraints=args.curr_constraints,
        confounders=args.confounders,
        target=args.target,
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
                    f"Duplicated trial: {trial.params}, return {previous_trial.value}"
                )
                # load args for previous trial and then save them for this trial
                shutil.copyfile(
                    f"{save_folder}/args_{previous_trial.number}.txt",
                    f"{save_folder}/args_{trial.number}.txt",
                )
                return previous_trial.value

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

        x, _, _, _, _, _ = dataset[0]
        input_dim = x.shape[0]

        # .dataset.dataset accesses a Subset, then the original Dataset used to form the subset. This has the prior_fn_loc attribute
        vae = VanillaVAE(
            layer_width=args.layer_width,
            n_layers=args.n_layers,
            z_dim=args.z_dim,
            input_dim=input_dim,
            dropout=args.dropout,
            use_cuda=args.cuda,
        )

        # PyTorch 2 compiler
        vae = torch.compile(vae)

        optim = torch.optim.AdamW(
            params=vae.parameters(),
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
        patience = 30
        for epoch in range(0, args.num_epochs):
            # TRAINING
            vae.train()
            # compute number of batches for an epoch
            x_s_batches = len(data_loaders["train_x_s"])
            x_only_batches = len(data_loaders["train_x_only"])
            batches_per_epoch = x_s_batches + x_only_batches
            period_x_s_batches = int(batches_per_epoch / x_s_batches)

            # initialize variables to store loss values
            # Only one x loss since using vanilla VAE
            epoch_losses_x = 0.0
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
                        # Only need to move x to cuda
                        xs, ss, cs, ys, idxs, sts = (
                            xs.cuda(non_blocking=True),
                            ss,
                            cs,
                            ys,
                            idxs,
                            sts,
                        )

                    # Vanilla VAE so unsup regardless of availability of s
                    if args.cuda:
                        with torch.cuda.amp.autocast():
                            loss = vae.unsup(xs)
                        epoch_losses_x += loss.detach().item() / batches_per_epoch
                    else:
                        with torch.amp.autocast(
                            device_type="cpu", dtype=torch.bfloat16
                        ):
                            loss = vae.unsup(xs)
                        epoch_losses_x += loss.detach().item() / batches_per_epoch

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
                    vae.eval()

                    val_iter = iter(data_loaders["val"])
                    for i in range(len(data_loaders["val"])):
                        (xv, sv, cv, yv, idxv, stv) = next(val_iter)
                        if args.cuda:
                            # Only need to move x to cuda
                            xv, sv, yv, idxv, stv = (
                                xv.cuda(non_blocking=True),
                                sv,
                                yv,
                                idxv,
                                stv,
                            )

                        val_recon = vae.recon_error(xv)
                        val_loss = vae.unsup(xv)
                        epoch_losses_val += val_loss.detach().item() / val_batches
                        epoch_losses_val_recon += (
                            val_recon.detach().item() / val_batches
                        )

                metric_dict_list.append(
                    {
                        "epoch": epoch,
                        "x_loss": epoch_losses_x,
                        "val_loss": epoch_losses_val,
                        "val_recon_loss": epoch_losses_val_recon,
                    }
                )

                # Printing the epoch results in a neat table format
                print(f"\n[Seed {args.seed} Fold {curr_fold} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Loss:':<30}{epoch_losses_x:.4f}")
                print(f"{'Val Loss:':<30}{epoch_losses_val:.4f}")
                print(f"{'Val Recon Loss:':<30}{epoch_losses_val_recon:.4f}")
                print(f"{'-' * 50}")

                # EARLY STOPPING COUNTER
                if best_score is None:
                    best_score = epoch_losses_val
                else:
                    # Check if val_loss improves or not
                    if epoch_losses_val < best_score:
                        # val_loss improves, we update the latest best_score
                        # and save the current model
                        best_score = epoch_losses_val
                        print("Saving current model...")
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
                        "x_loss": epoch_losses_x,
                        "val_loss": np.nan,
                        "val_recon_loss": np.nan,
                    }
                )

                print(f"\n[Seed {args.seed} Epoch {epoch}]")
                print(f"{'Metrics:':<30}")
                print(f"{'-' * 50}")
                print(f"{'Metric':<30}{'Value'}")
                print(f"{'-' * 50}")
                print(f"{'Loss:':<30}{epoch_losses_x:.4f}")
                print(f"{'-' * 50}")

        if hopt:
            # SAVE LOSSES FROM FOLD TO CSV
            losses_df = pd.DataFrame.from_dict(metric_dict_list)
            losses_df.to_csv(f"{save_folder}/losses_{trial.number}_{curr_fold}.csv")

            # GET BEST EPOCH
            best_epoch = losses_df["val_loss"].idxmin() + 1

            # RETURN FOR HYPEROPT
            # filter for unstable training points, e.g. train loss >2x val loss
            losses_df["unstable"] = losses_df["x_loss"] > 2 * losses_df["val_loss"]
            losses_df_report = losses_df[losses_df["unstable"] == False]
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
            json.dump(args.__dict__, f, indent=2)

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
            {"state_dict": vae.state_dict()},
            f"{save_folder}/best_model_{args.seed}.tar",
        )
        vae.save_models(save_folder, seed=args.seed)

        # SAVE BEST ARGS
        with open(f"{save_folder}/args_best.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)


def parser_args(parser):
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        help="Use GPU(s) for training; requires a CUDA-enabled PyTorch install.",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        default=300,
        type=int,
        help="Maximum number of training epochs per fold (--num-epochs / -n). Early stopping may terminate sooner.",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default="depmap_gdsc",
        metavar="D",
        help=(
            "Dataset name; see src/utils/data_utils.py:process_data for registered "
            "options (e.g. 'depmap_gdsc', 'depmap_gdsc_transneo')."
        ),
    )
    # Fixed seed for hopt
    parser.add_argument(
        "-seed",
        default=4563,
        type=int,
        help=(
            "Random seed. Used both for the Optuna TPE sampler and for Python/NumPy/PyTorch "
            "random number generation inside the model."
        ),
    )
    # Fixed arguments for hyperparameter optimisation -- use defaults
    parser.add_argument(
        "-target",
        type=str,
        default=None,
        metavar="D",
        help=(
            "Target column in y (e.g. 'AFATINIB' for depmap_gdsc, 'RCB.score' for "
            "TransNEO treatment response)."
        ),
    )
    parser.add_argument(
        "-constraints",
        default=None,
        type=str,
        nargs="+",
        help=(
            "If specified, uses these constraints rather than using defaults or "
            "selecting automatically."
        ),
    )
    parser.add_argument(
        "--confounders",
        default=None,
        type=str,
        nargs="+",
        help="Confounders used in final prediction model.",
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
        "--newstudy",
        default=False,
        action="store_true",
        help=(
            "Delete the existing Optuna SQLite study at the target save folder before "
            "creating a fresh one. Use this when you've changed the hyperparameter search space."
        ),
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help=(
            "Enqueue the single best trial from a prior AFATINIB hopt study (loaded from "
            "the matching 'AFATINIB/<experiment>/vae' folder) and run only that trial. "
            "A shortcut for per-drug refitting once the AFATINIB hopt has been done."
        ),
    )
    return parser


if __name__ == "__main__":
    # 1. Parse args and build save folder
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    if args.experiment is not None:
        save_folder = (
            f"{wd_path}/data/outputs/{args.dataset}/{args.target}/{args.experiment}/vae"
        )
    else:
        save_folder = f"{wd_path}/data/outputs/{args.dataset}/{args.target}/default/vae"

    if args.lindec:
        # If using a linear decoder
        save_folder = f"{save_folder}_ld"
        icovae_save_folder = f"{save_folder}_ld"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Put save folder in args
    args.save_folder = save_folder

    # 2. Load data and select constraints
    # TASK SPECIFIC SECTION -- EDIT BELOW HERE
    x, s, c, y, test_samples = process_data(
        dataset=args.dataset, wd_path=wd_path, experiment=args.experiment
    )
    args.test_samples = test_samples
    # END OF TASK SPECIFIC SECTION -- DO NOT EDIT BELOW HERE
    # SELECTING CONSTRAINTS
    if args.constraints is None:
        # Loading constraints for depmap_gdsc
        args.constraints = get_constraints(
            drug=args.target,
            dataset_name=args.dataset,
            zdim=512,
            experiment=args.experiment,
            col_thresh=0.9,
            wd_path=wd_path,
        )
    else:
        if args.constraints[0] == "breast_drivers":
            args.constraints = np.loadtxt(
                f"{wd_path}/data/processed/datasets/breast_cancer_drivers_filt.txt",
                dtype="str",
            ).tolist()
        else:
            pass
    # SET OTHER ARGUMENTS
    # iCoVAE stage
    args.stage = "i"

    # 3. Build Optuna study
    # CREATE OPTUNA STUDY
    # Create Optuna study with 50 random startup trials

    # If specified to start new study and the study already exists, delete the study
    if args.newstudy:
        try:
            optuna.delete_study(
                storage=f"sqlite:////{save_folder}/vae_optuna.db",
                study_name="_".join(save_folder.split("/")[-3:]),
            )
        except:
            pass

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50, multivariate=True, seed=args.seed
        ),
        storage=f"sqlite:////{save_folder}/vae_optuna.db",
        study_name="_".join(save_folder.split("/")[-3:]),
        load_if_exists=True,
    )
    # RECOMMENDED PARAMETERS FOR TRIAL USE
    # LOAD PARAMETERS FROM ICOVAE STUDY

    n_complete_trials = len(study.trials)

    if args.test:
        # Load best trial from AFATINIB experiment
        prev_save_folder = (
            f"{wd_path}/data/outputs/{args.dataset}/AFATINIB/{args.experiment}/vae"
        )
        prev_study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=50, multivariate=True, seed=args.seed
            ),
            storage=f"sqlite:////{prev_save_folder}/vae_optuna.db",
            study_name="_".join(prev_save_folder.split("/")[-3:]),
            load_if_exists=True,
        )
        study.enqueue_trial(prev_study.best_trial.params)
        print(f"Queueing trial: {prev_study.best_trial.params}")
        n_trials = 1
    else:
        n_trials = 150

    # 4. Run hopt
    # HYPERPARAMETER OPTIMIZATION FOR 1 TRIAL
    # Just run this to get the number of epochs to train for
    func = lambda trial: main(
        trial, x=x, s=s, c=c, y=y, args=args, save_folder=save_folder
    )
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

    # 5. Refit best trial for 10 seeds
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
