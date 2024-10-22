import torch
from torch.nn import functional as F


def train_epoch(model, optimizer, device, data_loader):

    model.train()
    epoch_train_loss = 0
    nb_data = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        if batch_graphs.number_of_edges() == 0:
            iter -= 1
            continue
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata["feat"].to(device)
        batch_e = batch_graphs.edata["feat"].to(device)
        batch_targets = batch_targets.to(device)
        batch_targets, batch_rels, batch_id = torch.split(batch_targets, 1, dim=1)

        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)

        loss = model.loss(batch_scores, batch_targets, batch_rels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.detach().item()

        nb_data += batch_targets.size(0)

    epoch_train_loss /= iter + 1

    return epoch_train_loss, optimizer


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"].to(device)
            batch_e = batch_graphs.edata["feat"].to(device)
            batch_targets = batch_targets.to(device)
            batch_targets, batch_rels, batch_id = torch.split(batch_targets, 1, dim=1)

            try:
                batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)

            loss = F.mse_loss(batch_scores, batch_targets)

            epoch_test_loss += loss.detach().item()

            nb_data += batch_targets.size(0)
        epoch_test_loss /= iter + 1

        return epoch_test_loss
