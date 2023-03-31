import numpy as np
import torch

# def map(submission_array, gt_array):
#     """ Returns mAP, weighted mAP, and AP array """
#     m_aps = []
#     n_classes = submission_array.shape[1]
#     for oc_i in range(n_classes):
#         sorted_idxs = torch.argsort(-submission_array[:, oc_i])
#         tp = gt_array[:, oc_i][sorted_idxs] == 1       #高到低排序，正样本转化为True，负样本转化为False
#         fp = torch.logical_not(tp)               #模型预测为负类但实际上是正类的样本，假阳
#         n_pos = tp.sum()                        #计算正例的个数，即在gt_array中标记为1的数量
#         if n_pos < 0.1:
#             m_aps.append(float('nan'))
#             continue
#         fp.sum()
#         f_pcs = torch.cumsum(fp,dim=0)
#         t_pcs = torch.cumsum(tp,dim=0)
#         prec = t_pcs / (f_pcs + t_pcs).float()
#         avg_prec = 0
#         for i in range(submission_array.shape[0]):
#             if tp[i]:
#                 # avg_prec += prec[i]
#                 avg_prec = prec[i]+avg_prec
#         m_aps.append(avg_prec / n_pos.float())
#     m_aps = torch.tensor(m_aps)
#     # m_ap = torch.mean(m_aps)
#     mask = torch.isnan(m_aps)
#     m_aps[mask] = 0
#     m_ap = torch.mean(m_aps)
#     gt_array = gt_array.to(m_aps.device)
#     w_ap = (m_aps * gt_array.sum(dim=0) / gt_array.sum().sum().float())

#     return m_ap, w_ap, m_aps

def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    device = submission_array.device
    for oc_i in range(n_classes):
        sorted_idxs = torch.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1       #高到低排序，正样本转化为True，负样本转化为False
        fp = torch.logical_not(tp)               #模型预测为负类但实际上是正类的样本，假阳
        n_pos = tp.sum()                        #计算正例的个数，即在gt_array中标记为1的数量
        if n_pos < 0.1:
            m_aps.append(torch.tensor(float('nan'), device=device))
            continue
        fp.sum()
        f_pcs = torch.cumsum(fp, dim=0)
        t_pcs = torch.cumsum(tp, dim=0)
        prec = t_pcs / (f_pcs + t_pcs).float()
        avg_prec = torch.tensor(0., device=device)
        for i in range(submission_array.shape[0]):
            if tp[i]:
                # avg_prec += prec[i]
                avg_prec = prec[i]+avg_prec
        m_aps.append(avg_prec / n_pos.float())
    m_aps = torch.stack(m_aps)
    mask = torch.isnan(m_aps)
    m_aps[mask] = 0.
    m_ap = m_aps.mean()
    gt_array = gt_array.to(device)
    w_ap = (m_aps * gt_array.sum(dim=0) / gt_array.sum().sum().float())
    
    return m_ap, w_ap, m_aps




def charades_map(submission_array, gt_array):
    """ 
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    # fix = submission_array.copy()
    # empty = np.sum(gt_array, axis=1)==0
    # fix[empty, :] = np.NINF
    # return map(fix, gt_array)

    fix = submission_array.to('cuda')
    gt_array = gt_array.to('cuda')
    empty = torch.sum(gt_array, dim=1) == 0
    fix[empty, :] = float('-inf')
    
    return map(fix, gt_array)