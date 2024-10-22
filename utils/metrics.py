from math import isnan

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


class SquaredErrorRelevanceArea:

    def __init__(self, n=100, device=torch.device("cuda")):

        # Creating a list from 1 to 2 with 0.01 for intervals for increasing relevances
        intervals = np.arange(1.0, 2 + 1.5 / n, 1.0 / n)

        self.intervals = torch.tensor(intervals, device=device)

    def loss(self, y_pred, y_true, y_rel=None):

        expanded_inter = self.intervals.expand(len(y_true), len(self.intervals))
        expanded_rel = y_rel.T.expand(len(self.intervals), len(y_true)).T

        squared_errors = (y_true - y_pred) ** 2
        mask = expanded_rel >= expanded_inter

        ser = torch.matmul(squared_errors.T, mask.type(torch.float32))
        sera = torch.sum(torch.cumsum(ser, dim=1)) / y_pred.size(0)

        return sera


class LossComparison:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        model.eval()

    def by_target_type(self, data_loader, theset):
        MSE_general = []
        MSEs_on_zero = []
        MSEs_on_nonzero = []

        all_scores = []
        all_targets = []
        all_output = []

        with torch.no_grad():
            for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

                batch_graphs = batch_graphs.to(self.device)
                batch_x = batch_graphs.ndata["feat"].to(self.device)
                batch_e = batch_graphs.edata["feat"].to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_targets, batch_rels, batch_id = torch.split(batch_targets, 1, dim=1)

                try:
                    batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(self.device)
                except:
                    batch_lap_pos_enc = None

                try:
                    batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(self.device)
                except:
                    batch_wl_pos_enc = None

                batch_scores = self.model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)

                # Filter zero from non-zero targets and respective scores to compute MSEs separately
                mask = batch_targets == 0
                zero_indices, _ = torch.split(mask.nonzero(), 1, dim=1)

                mask = batch_targets > 0
                nonzero_indices, _ = torch.split(mask.nonzero(), 1, dim=1)

                batch_targets_z = torch.index_select(batch_targets, 0, zero_indices.squeeze())
                batch_targets_nz = torch.index_select(batch_targets, 0, nonzero_indices.squeeze())

                batch_scores_z = torch.index_select(batch_scores, 0, zero_indices.squeeze())
                batch_scores_nz = torch.index_select(batch_scores, 0, nonzero_indices.squeeze())

                targets_lst = [batch_targets_z, batch_targets_nz]
                scores_lst = [batch_scores_z, batch_scores_nz]

                # Compute the two MSEs
                for idx, (t, s) in enumerate(zip(targets_lst, scores_lst)):
                    epoch_mse = F.mse_loss(s, t).detach().item()
                    MSE_general.append(epoch_mse)
                    if idx == 0:
                        MSEs_on_zero.append(epoch_mse)
                    else:
                        MSEs_on_nonzero.append(epoch_mse)

                # For the scatter plot
                all_scores.append(batch_scores.squeeze().cpu().detach().numpy())
                all_targets.append(batch_targets.squeeze().cpu().detach().numpy())

                # For export (pickle and csv)
                for t, s, theid in zip(
                    batch_targets.squeeze().cpu().detach().numpy(),
                    batch_scores.squeeze().cpu().detach().numpy(),
                    batch_id.squeeze().cpu().detach().numpy(),
                ):
                    all_output.append((t, s, int(theid)))

        def scatter():

            fig, ax = plt.subplots()
            high, med, low, zero = [], [], [], []
            low_thold = 0.1
            med_thold = 0.2

            for tar, sco in zip(all_targets, all_scores):
                for t, s in zip(tar, sco):

                    if t > med_thold:
                        high.append((t, s))
                    elif t > low_thold:
                        med.append((t, s))
                    elif t > 0:
                        low.append((t, s))
                    else:
                        zero.append((t, s))

            # Subplot for zero targets
            x, y = zip(*zero)
            ax.scatter(x, y, c="yellow", label="Zero")

            # Subplot for targets up to low_thold
            x, y = zip(*low)
            ax.scatter(x, y, c="blue", label="Low positive")

            # Subplot for targets up to medthold
            x, y = zip(*med)
            ax.scatter(x, y, c="red", label="Medium positive")

            # Subplot for targets above med_thold
            x, y = zip(*high)
            ax.scatter(x, y, c="black", label="High positive")

            ax.set_title("Scatter plot", fontsize=16)

            plt.xlabel("True target", fontsize=14)
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
            ax.tick_params(axis="x", labelsize=16)

            plt.ylabel("Predicted target", fontsize=14)
            plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
            ax.tick_params(axis="y", labelsize=16)

            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])

            # y=x line
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black")

            # Threshold lines
            plt.axvline(x=0, c="blue", linestyle="--")
            plt.axvline(x=low_thold, c="red", linestyle="--")
            plt.axvline(x=med_thold, c="black", linestyle="--")

            ax.grid(True)
            plt.legend(loc=2, prop={"size": 20})

            # plt.show()

        if theset == "testing set":
            scatter()

        MSE_general = [val for val in MSE_general if isnan(val) == False]
        MSEs_on_zero = [val for val in MSEs_on_zero if isnan(val) == False]
        MSEs_on_nonzero = [val for val in MSEs_on_nonzero if isnan(val) == False]

        MSE_general = sum(MSE_general) / len(MSE_general)
        bulk_MSE = sum(MSEs_on_zero) / len(MSEs_on_zero)
        extremes_MSE = sum(MSEs_on_nonzero) / len(MSEs_on_nonzero)

        return all_output, MSE_general, bulk_MSE, extremes_MSE
