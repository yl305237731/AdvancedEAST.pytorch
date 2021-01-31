import torch
from torch import nn


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-4

    def quad_norm(self, g_true):
        shape = g_true.shape
        delta_xy_matrix = g_true.reshape([-1, 2, 2])
        diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
        square = torch.square(diff)
        distance = torch.sqrt(torch.sum(square, dim=-1))
        distance *= 4.0
        distance += self.epsilon
        return distance.reshape(shape[:-1])

    def forward(self, netout, gt, weights):
        n_q = torch.reshape(self.quad_norm(gt), weights.shape)
        diff = netout - gt
        abs_diff = torch.abs(diff)
        mask = torch.lt(abs_diff, 1).type(torch.float)
        pixel_wise_loss = ((mask * 0.5 * torch.square(abs_diff) + (1 - mask) * abs_diff - 0.5).sum(
            dim=-1) / n_q) * weights
        return pixel_wise_loss


class QUADLoss(nn.Module):
    def __init__(self, epsilon=1e-4, lambda_inside_score_loss=4.0, lambda_side_vertex_code_loss=1.0,
                 lambda_side_vertex_coord_loss=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.lambda_inside_score_loss = lambda_inside_score_loss
        self.lambda_side_vertex_code_loss = lambda_side_vertex_code_loss
        self.lambda_side_vertex_coord_loss = lambda_side_vertex_coord_loss
        self.smooth_l1_loss = SmoothL1Loss()

    def forward(self, y_pred, y_true):
        # loss for inside_score
        logits = y_pred[:, :, :, :1]
        labels = y_true[:, :, :, :1]
        beta = 1 - torch.mean(labels)
        predicts = torch.sigmoid(logits)
        inside_score_loss = torch.mean(-1 * (beta * labels * torch.log(predicts + self.epsilon) +
                                             (1 - beta) * (1 - labels) * torch.log(1 - predicts + self.epsilon)))
        inside_score_loss *= self.lambda_inside_score_loss

        vertex_logits = y_pred[:, :, :, 1:3]
        vertex_labels = y_true[:, :, :, 1:3]
        vertex_beta = 1 - (torch.mean(y_true[:, :, :, 1:2]) / (torch.mean(labels) + self.epsilon))
        vertex_predicts = torch.sigmoid(vertex_logits)
        pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts + self.epsilon)
        neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(1 - vertex_predicts + self.epsilon)
        positive_weights = torch.eq(y_true[:, :, :, 0], 1).type(torch.float)
        side_vertex_code_loss = torch.sum(torch.sum(pos + neg, dim=-1) * positive_weights) / (
                    torch.sum(positive_weights) + self.epsilon)
        side_vertex_code_loss *= self.lambda_side_vertex_code_loss

        # loss for side_vertex_coord delta
        g_hat = y_pred[:, :, :, 3:]
        g_true = y_true[:, :, :, 3:]
        vertex_weights = torch.eq(y_true[:, :, :, 1], 1).type(torch.float)
        pixel_wise_smooth_l1norm = self.smooth_l1_loss(g_hat, g_true, vertex_weights)
        side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (torch.sum(vertex_weights) + self.epsilon)
        side_vertex_coord_loss *= self.lambda_side_vertex_coord_loss
        return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss
