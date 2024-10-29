import json
import numpy as np
import torch
from torch import nn
from jigsaw.losses import EOLossYoto


def compute_dp_regularized_loss_and_acc_per_batch_single_lambda(
    model, batch, device, lambda_lb, lambda_ub, square_dp
):
    data, labels, sensitive_attr = batch
    data, labels, sensitive_attr = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
    )
    sensitive_0 = (sensitive_attr == 0).reshape(-1)
    sensitive_1 = (sensitive_attr == 1).reshape(-1)
    if sensitive_0.sum() == 0 or sensitive_1.sum() == 0:
        return None, None, None
    lambda_reg = (
        torch.exp(torch.FloatTensor(1).uniform_(np.log(lambda_lb), np.log(lambda_ub)))
        .reshape(-1, 1)
        .to(device)
    )
    outputs = model(data, lambda_reg)
    if square_dp:
        dp_loss = torch.square(
            torch.mean(outputs[sensitive_0, :]) - torch.mean(outputs[sensitive_1, :])
        )
    else:
        dp_loss = torch.abs(
            torch.mean(outputs[sensitive_0, :]) - torch.mean(outputs[sensitive_1, :])
        )
    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels)
    accuracy = (
        (
            outputs.reshape(-1) * labels.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
        )
        .sum()
        .item()
    )
    return (dp_loss, classification_loss + lambda_reg.reshape(-1) * dp_loss, accuracy)


def compute_eo_regularized_loss_and_acc_per_batch_single_lambda(
    model, batch, device, lambda_lb, lambda_ub, square_eo
):
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = batch
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
        X_y0_s0.to(device),
        X_y0_s1.to(device),
        X_y1_s0.to(device),
        X_y1_s1.to(device),
    )

    lambda_reg = (
        torch.exp(torch.FloatTensor(1).uniform_(np.log(lambda_lb), np.log(lambda_ub)))
        .reshape(-1, 1)
        .to(device)
    )

    outputs = model(data, lambda_reg.repeat(data.shape[0], 1))

    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels)

    # True positive rates for each group
    tpr0 = model(X_y1_s0, lambda_reg).mean()
    tpr1 = model(X_y1_s1, lambda_reg).mean()

    # False positive rates for each group
    fpr0 = model(X_y0_s0, lambda_reg).mean()
    fpr1 = model(X_y0_s1, lambda_reg).mean()

    if square_eo:
        eo_loss = torch.square(tpr0 - tpr1) + torch.square(fpr0 - fpr1)
    else:
        eo_loss = torch.abs(tpr0 - tpr1) + torch.abs(fpr0 - fpr1)

    combined_loss = classification_loss + lambda_reg.reshape(-1) * eo_loss
    acc = (
        (
            outputs.reshape(-1) * labels.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
        )
        .sum()
        .item()
    )
    return (eo_loss, combined_loss, acc)


def compute_eop_regularized_loss_and_acc_per_batch_single_lambda(
    model, batch, device, lambda_lb, lambda_ub, square_eop
):
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = batch
    data, labels, sensitive_attr, X_y0_s0, X_y0_s1, X_y1_s0, X_y1_s1 = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
        X_y0_s0.to(device),
        X_y0_s1.to(device),
        X_y1_s0.to(device),
        X_y1_s1.to(device),
    )

    lambda_reg = (
        torch.exp(torch.FloatTensor(1).uniform_(np.log(lambda_lb), np.log(lambda_ub)))
        .reshape(-1, 1)
        .to(device)
    )

    outputs = model(data, lambda_reg.repeat(data.shape[0], 1))

    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels)

    # True positive rates for each group
    tpr0 = model(X_y1_s0, lambda_reg).mean()
    tpr1 = model(X_y1_s1, lambda_reg).mean()

    if square_eop:
        eop_loss = torch.square(tpr0 - tpr1)
    else:
        eop_loss = torch.abs(tpr0 - tpr1)

    combined_loss = classification_loss + lambda_reg.reshape(-1) * eop_loss
    acc = (
        (
            outputs.reshape(-1) * labels.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels).reshape(-1))
        )
        .sum()
        .item()
    )
    return (eop_loss, combined_loss, acc)


def compute_dp_regularized_loss_and_acc_per_batch(
    model, batch, device, lambda_lb, lambda_ub, square_dp
):
    data, labels, sensitive_attr = batch
    data, labels, sensitive_attr = (
        data.to(device),
        labels.to(device),
        sensitive_attr.to(device),
    )
    sensitive_0 = (sensitive_attr == 0).reshape(-1)
    sensitive_1 = (sensitive_attr == 1).reshape(-1)
    if sensitive_0.sum() == 0 or sensitive_1.sum() == 0:
        return None, None, None
    data_0, data_1 = data[sensitive_0, :], data[sensitive_1, :]
    lambda_reg = (
        torch.exp(
            torch.FloatTensor(data.shape[0]).uniform_(
                np.log(lambda_lb), np.log(lambda_ub)
            )
        )
        .reshape(-1, 1)
        .to(device)
    )
    lambda_reg0 = lambda_reg[sensitive_0, :]
    lambda_reg1 = lambda_reg[sensitive_1, :]
    outputs_0 = model(data_0, lambda_reg0)
    outputs_1 = model(data_1, lambda_reg1)
    if square_dp:
        dp_loss = torch.square(
            torch.mean(lambda_reg0 * outputs_0) - torch.mean(lambda_reg1 * outputs_1)
        )
    else:
        dp_loss = torch.abs(
            torch.mean(lambda_reg0 * outputs_0) - torch.mean(lambda_reg1 * outputs_1)
        )
    outputs = torch.cat([outputs_0, outputs_1], dim=0)
    labels_reordered = torch.cat(
        [labels[sensitive_0, :], labels[sensitive_1, :]], dim=0
    )
    bce_loss = nn.BCELoss()
    classification_loss = bce_loss(outputs, labels_reordered)
    accuracy = (
        (
            outputs.reshape(-1) * labels_reordered.reshape(-1)
            + ((1 - outputs).reshape(-1) * (1 - labels_reordered).reshape(-1))
        )
        .sum()
        .item()
    )
    return (dp_loss, classification_loss + dp_loss, accuracy)


def get_loss_function(input_dim, fairness, per_batch_lambda, train_dataset):
    if input_dim == 512 and fairness.startswith("eo"):
        loss_per_batch = EOLossYoto(
                fairness=fairness, train_df=train_dataset
            ).loss_function
    elif fairness == "eo":
        loss_per_batch = compute_eo_regularized_loss_and_acc_per_batch_single_lambda
    elif fairness == "eop":
        loss_per_batch = compute_eop_regularized_loss_and_acc_per_batch_single_lambda
    elif per_batch_lambda:
        loss_per_batch = compute_dp_regularized_loss_and_acc_per_batch_single_lambda
    else:
        loss_per_batch = compute_dp_regularized_loss_and_acc_per_batch
    return loss_per_batch

