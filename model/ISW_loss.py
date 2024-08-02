# import pdb
#
# import torch
#
# eps = 1e-5
#
#
# def get_covariance_matrix(f_map):
#     B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
#     HW = H * W
#     eye = torch.eye(C).cuda()
#     f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
#     f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW
#
#     return f_cor
#
#
# def ISWloss(query1, query2, relax_denom=2):
#
#     B, C, H, W = query1.shape
#     HW = H * W
#     eye = torch.eye(C).cuda()
#     mask_matrix = torch.triu(torch.ones(C, C), diagonal=1).cuda()
#     num_off_diagonal = torch.sum(mask_matrix)
#     margin = num_off_diagonal / relax_denom
#     query1 = query1.contiguous().view(B, C, -1)
#     query2 = query2.contiguous().view(B, C, -1)
#     f_cov1 = torch.bmm(query1, query1.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW
#     f_cov2 = torch.bmm(query2, query2.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW
#
#     cov_mean = 0.5*(f_cov1 + f_cov2)
#     cov_var = 0.5 * ((f_cov1 - cov_mean)**2 + (f_cov2 - cov_mean)**2)
#     var_flatten = torch.flatten(cov_var)
#     num_sensitive = num_off_diagonal - margin
#     _, indices = torch.topk(var_flatten, k=int(num_sensitive))
#     mask_matrix = torch.flatten(torch.zeros(B, C, C).cuda())
#     mask_matrix[indices] = 1
#     #pdb.set_trace()
#     mask_matrix = mask_matrix.view(B, C, C)
#     mask_matrix = 1 - mask_matrix
#     mask_matrix = torch.triu(mask_matrix, diagonal = 0)
#
#     f_cov_masked1 = f_cov1 * mask_matrix
#     f_cov_masked2 = f_cov2 * mask_matrix
#
#     off_diag_sum = torch.sum(torch.abs(f_cov_masked1), dim=(1, 2), keepdim=True)  # - margin  # B X 1 X 1
#     loss1 = torch.clamp(torch.div(off_diag_sum, num_sensitive), min=0)  # B X 1 X 1
#     off_diag_sum = torch.sum(torch.abs(f_cov_masked2), dim=(1, 2), keepdim=True)  # - margin  # B X 1 X 1
#     loss2 = torch.clamp(torch.div(off_diag_sum, num_sensitive), min=0)  # B X 1 X 1
#
#     return (torch.sum(loss1) + torch.sum(loss2)) / B


import pdb

import torch
import kmeans1d
eps = 1e-5


def get_covariance_matrix(f_map):
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor


def ISWloss(query1, query2, relax_denom=2, numclusters=50):

    B, C, H, W = query1.shape
    HW = H * W
    eye = torch.eye(C).cuda()
    mask_matrix = torch.triu(torch.ones(C, C), diagonal=1).cuda()
    num_off_diagonal = torch.sum(mask_matrix)
    margin = num_off_diagonal / relax_denom
    query1 = query1.contiguous().view(B, C, -1)
    query2 = query2.contiguous().view(B, C, -1)
    f_cov1 = torch.bmm(query1, query1.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW
    f_cov2 = torch.bmm(query2, query2.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

    cov_mean = 0.5*(f_cov1 + f_cov2)
    cov_var = 0.5 * ((f_cov1 - cov_mean)**2 + (f_cov2 - cov_mean)**2)
    var_flatten = torch.flatten(cov_var)
    clusters, centroids = kmeans1d.cluster(var_flatten, numclusters)  # 50 clusters
    num_sensitive = var_flatten.size()[0] - clusters.count(0)  # 1: Insensitive Cov, 2~50: Sensitive Cov
    _, indices = torch.topk(var_flatten, k=int(num_sensitive))
    mask_matrix = torch.flatten(torch.zeros(B, C, C).cuda())
    mask_matrix[indices] = 1
    #pdb.set_trace()
    mask_matrix = mask_matrix.view(B, C, C)
    mask_matrix = 1 - mask_matrix
    mask_matrix = torch.triu(mask_matrix, diagonal = 0)
    num_mask = mask_matrix.sum()
    f_cov_masked1 = f_cov1 * mask_matrix
    f_cov_masked2 = f_cov2 * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cov_masked1), dim=(1, 2), keepdim=True)  # - margin  # B X 1 X 1
    loss1 = torch.clamp(torch.div(off_diag_sum, num_mask), min=0)  # B X 1 X 1
    off_diag_sum = torch.sum(torch.abs(f_cov_masked2), dim=(1, 2), keepdim=True)  # - margin  # B X 1 X 1
    loss2 = torch.clamp(torch.div(off_diag_sum, num_mask), min=0)  # B X 1 X 1

    return (torch.sum(loss1) + torch.sum(loss2)) / B