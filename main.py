import argparse
import glob
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from data.load_data import LoadData
from nets.load_net import gnn_model
from train.train import evaluate_network, train_epoch
from utils.early_stop import EarlyStopping
from utils.loss_vis import visualise_losses
from utils.metrics import LossComparison

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        print("cuda not available")
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(model_name, net_params):
    model = gnn_model(model_name, net_params)
    total_param = 0
    print("\nMODEL DETAILS:")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(model_name + "," + " Total parameters:", total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(model_name, dataset, params, net_params, dirs):

    start0 = time.time()
    per_epoch_time = []

    dataset_name = dataset.name

    print("\nTraining Graphs: ", len(dataset.train))
    print("Validation Graphs: ", len(dataset.val))
    print("Test Graphs: ", len(dataset.test))

    if net_params["lap_pos_enc"]:
        # print("\n[I] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(net_params["pos_enc_dim"])

    if net_params["wl_pos_enc"]:
        # print("\n[I] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()

    print("\n[I] Initiating training...\n")

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params["device"]

    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + ".txt", "w") as f:
        f.write(
            """Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
                dataset_name, model_name, params, net_params, net_params["total_param"]
            )
        )

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(params["seed"])

    model = gnn_model(model_name, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["init_lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=params["lr_reduce_factor"], patience=params["lr_schedule_patience"], verbose=True
    )

    train_loader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params["batch_size"], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, collate_fn=dataset.collate)

    epoch_train_Losses, epoch_val_Losses = [], []
    last_train_Losses, last_val_Losses = [], []
    # Create an instance of EarlyStopping
    early_stopping = EarlyStopping()

    try:

        with tqdm(range(params["epochs"])) as t:

            for epoch in t:

                t.set_description(f"Epoch {epoch}")

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, train_loader)
                epoch_val_loss = evaluate_network(model, device, val_loader)

                epoch_train_Losses.append(epoch_train_loss)
                epoch_val_Losses.append(epoch_val_loss)

                writer.add_scalar("train/_loss", epoch_train_loss, epoch)
                writer.add_scalar("val/_loss", epoch_val_loss, epoch)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

                t.set_postfix(
                    time=time.time() - start,
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                )

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), "{}.pkl".format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + "/*.pkl")
                for file in files:
                    epoch_nb = file.split("_")[-1]
                    epoch_nb = int(epoch_nb.split(".")[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]["lr"] < params["min_lr"]:
                    print("\n[!] Learning rate smaller or equal to min learning rate threshold, quitting.\n")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - start0 > params["max_time"] * 3600:
                    print("-" * 89)
                    print("\n[!] Max_time for training elapsed {:.2f} hours, quitting.\n".format(params["max_time"]))
                    break

                # Checking for overfitting every N epochs (early stopping)
                last_train_Losses.append(epoch_train_loss)
                last_val_Losses.append(epoch_val_loss)

                if epoch >= 79 and (epoch + 1) % 80 == 0:
                    # early_stopping(last_train_Losses, last_val_Losses)
                    early_stopping(last_val_Losses)

                    if early_stopping.early_stop:
                        print("\n[!] Exiting early from training because of overfitting or loss plateau")
                        break

                if (epoch + 1) % 80 == 0:
                    # Reset last losses
                    last_train_Losses = []
                    last_val_Losses = []

    except KeyboardInterrupt:
        print("-" * 89)
        print("\n[!] Exiting early from training because of KeyboardInterrupt")

    # Visualise the learning curves
    epochs = np.arange(1, len(epoch_train_Losses) + 1)
    visualise_losses(epochs, epoch_train_Losses, epoch_val_Losses)

    # Compute the final MSEs for zero and positive targets
    MSE_comparison = LossComparison(model, device)
    train_all_output, train_mse_general, train_mse_bulk, train_mse_extremes = MSE_comparison.by_target_type(
        train_loader, "training set"
    )
    test_all_output, test_mse_general, test_mse_bulk, test_mse_extremes = MSE_comparison.by_target_type(
        test_loader, "testing set"
    )

    with open(f"out/SU/PredsOutSera/sera_preds_{dataset_name}.pkl", "wb") as f:
        pickle.dump(test_all_output, f)
    print("\nPredictions saved to output file")

    print(f"Test Loss - general: {test_mse_general} | on zeroes: {test_mse_bulk} | on positives: {test_mse_extremes}")
    print("\nConvergence Time (epochs): {:.4f}".format(epoch))
    print("Total time taken {:.4f}s".format(time.time() - start0))
    print("Avg time per epoch: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + ".txt", "w") as f:
        f.write(
            """Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTRAIN MSE (bulk): {:.4f}\nTRAIN MSE (extremes): {:.4f}\nTEST MSE (bulk): {:.4f}\nTEST MSE (extremes): {:.4f}\n
    Convergence Time (epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""".format(
                dataset_name,
                model_name,
                params,
                net_params,
                model,
                net_params["total_param"],
                train_mse_bulk,
                train_mse_extremes,
                test_mse_bulk,
                test_mse_extremes,
                epoch,
                (time.time() - start0) / 3600,
                np.mean(per_epoch_time),
            )
        )


def main():
    """
    USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Please give a config.json file with training/model/data/param details")
    parser.add_argument("--gpu_id", help="Please give a value for gpu id")
    parser.add_argument("--model", help="Please give a value for model name")
    parser.add_argument("--dataset", help="Please give a value for dataset name")
    parser.add_argument("--out_dir", help="Please give a value for out_dir")
    parser.add_argument("--seed", help="Please give a value for seed")
    parser.add_argument("--epochs", help="Please give a value for epochs")
    parser.add_argument("--batch_size", help="Please give a value for batch_size")
    parser.add_argument("--init_lr", help="Please give a value for init_lr")
    parser.add_argument("--lr_reduce_factor", help="Please give a value for lr_reduce_factor")
    parser.add_argument("--lr_schedule_patience", help="Please give a value for lr_schedule_patience")
    parser.add_argument("--min_lr", help="Please give a value for min_lr")
    parser.add_argument("--weight_decay", help="Please give a value for weight_decay")
    parser.add_argument("--print_epoch_interval", help="Please give a value for print_epoch_interval")
    parser.add_argument("--L", help="Please give a value for L")
    parser.add_argument("--hidden_dim", help="Please give a value for hidden_dim")
    parser.add_argument("--out_dim", help="Please give a value for out_dim")
    parser.add_argument("--residual", help="Please give a value for residual")
    parser.add_argument("--edge_feat", help="Please give a value for edge_feat")
    parser.add_argument("--readout", help="Please give a value for readout")
    parser.add_argument("--n_heads", help="Please give a value for n_heads")
    parser.add_argument("--in_feat_dropout", help="Please give a value for in_feat_dropout")
    parser.add_argument("--dropout", help="Please give a value for dropout")
    parser.add_argument("--layer_norm", help="Please give a value for layer_norm")
    parser.add_argument("--batch_norm", help="Please give a value for batch_norm")
    parser.add_argument("--self_loop", help="Please give a value for self_loop")
    parser.add_argument("--max_time", help="Please give a value for max_time")
    parser.add_argument("--pos_enc_dim", help="Please give a value for pos_enc_dim")
    parser.add_argument("--lap_pos_enc", help="Please give a value for lap_pos_enc")
    parser.add_argument("--wl_pos_enc", help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open("configs/SU_params.json") as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config["gpu"]["id"] = int(args.gpu_id)
        config["gpu"]["use"] = True
    device = gpu_setup(config["gpu"]["use"], config["gpu"]["id"])
    # model, dataset, out_dir
    if args.model is not None:
        model_name = args.model
    else:
        model_name = config["model"]

    dataset_file = config["dataset"]["pickle"]
    dataset = LoadData(dataset_file)

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config["out_dir"]

    # parameters
    params = config["params"]
    if args.seed is not None:
        params["seed"] = int(args.seed)
    if args.epochs is not None:
        params["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        params["batch_size"] = int(args.batch_size)
    if args.init_lr is not None:
        params["init_lr"] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params["lr_reduce_factor"] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params["lr_schedule_patience"] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params["min_lr"] = float(args.min_lr)
    if args.weight_decay is not None:
        params["weight_decay"] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params["print_epoch_interval"] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params["max_time"] = float(args.max_time)

    # network parameters
    net_params = config["net_params"]
    net_params["device"] = device
    net_params["gpu_id"] = config["gpu"]["id"]
    net_params["batch_size"] = params["batch_size"]
    if args.L is not None:
        net_params["L"] = int(args.L)
    if args.hidden_dim is not None:
        net_params["hidden_dim"] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params["out_dim"] = int(args.out_dim)
    if args.residual is not None:
        net_params["residual"] = True if args.residual == "True" else False
    if args.edge_feat is not None:
        net_params["edge_feat"] = True if args.edge_feat == "True" else False
    if args.readout is not None:
        net_params["readout"] = args.readout
    if args.n_heads is not None:
        net_params["n_heads"] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params["in_feat_dropout"] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params["dropout"] = float(args.dropout)
    if args.layer_norm is not None:
        net_params["layer_norm"] = True if args.layer_norm == "True" else False
    if args.batch_norm is not None:
        net_params["batch_norm"] = True if args.batch_norm == "True" else False
    if args.self_loop is not None:
        net_params["self_loop"] = True if args.self_loop == "True" else False
    if args.lap_pos_enc is not None:
        net_params["lap_pos_enc"] = True if args.pos_enc == "True" else False
    if args.pos_enc_dim is not None:
        net_params["pos_enc_dim"] = int(args.pos_enc_dim)
    if args.wl_pos_enc is not None:
        net_params["wl_pos_enc"] = True if args.pos_enc == "True" else False

    # Node and edge dimensions
    net_params["in_dim_node"] = dataset.train[0][0].ndata["feat"].size(1)
    try:
        net_params["in_dim_edge"] = dataset.train[0][0].edata["feat"].size(1)
    except KeyError:
        net_params["in_dim_edge"] = 2

    root_log_dir = (
        out_dir
        + "logs/"
        + model_name
        + "_"
        + dataset_file
        + "_GPU"
        + str(config["gpu"]["id"])
        + "_"
        + time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    )
    root_ckpt_dir = (
        out_dir
        + "checkpoints/"
        + model_name
        + "_"
        + dataset_file
        + "_GPU"
        + str(config["gpu"]["id"])
        + "_"
        + time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    )
    write_file_name = (
        out_dir
        + "results/result_"
        + model_name
        + "_"
        + dataset_file
        + "_GPU"
        + str(config["gpu"]["id"])
        + "_"
        + time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    )
    write_config_file = (
        out_dir
        + "configs/config_"
        + model_name
        + "_"
        + dataset_file
        + "_GPU"
        + str(config["gpu"]["id"])
        + "_"
        + time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    )
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + "results"):
        os.makedirs(out_dir + "results")

    if not os.path.exists(out_dir + "configs"):
        os.makedirs(out_dir + "configs")

    net_params["total_param"] = view_model_param(model_name, net_params)

    train_val_pipeline(model_name, dataset, params, net_params, dirs)


if __name__ == "__main__":
    main()
