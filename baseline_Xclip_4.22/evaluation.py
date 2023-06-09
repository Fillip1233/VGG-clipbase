import numpy as np


class Evaluator:
    def __init__(self, total_instances, total_classes) -> None:
        self.total_instances = total_instances
        self.total_classes = total_classes
        self.reset()
        self.best_mean_average_precision = 0.0

    def reset(self):
        self.index = 0
        self.predictions = np.zeros((self.total_instances, self.total_classes))
        self.ground_truths = np.zeros((self.total_instances, self.total_classes))

    def process(self, preds, labels):
        size = preds.shape[0]  # batch size
        self.predictions[self.index : self.index + size] = (
            preds.cpu().numpy()
        )
        self.ground_truths[self.index : self.index + size] = labels.cpu().numpy()
        self.index += size

    def evaluate(self):
        mean_average_precision, _, _ = charades_map(
            self.predictions, self.ground_truths
        )
        return {"map": mean_average_precision}

    def is_best(self):
        metrics = self.evaluate()
        if metrics["map"] > self.best_mean_average_precision:
            self.best_mean_average_precision = metrics["map"]
            return True
        return False
    

def map(submission_array, gt_array):
    # https://github.com/gsig/charades-algorithms/blob/master/pytorch/utils/map.py
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float("nan"))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.stack(m_aps)  
    mask = np.isnan(m_aps)
    m_aps[mask] = 0.
    m_ap = np.mean(m_aps)
    # m_aps = np.array(m_aps)
    # m_ap = np.mean(m_aps)
    w_ap = m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float)
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    # https://github.com/gsig/charades-algorithms/blob/master/pytorch/utils/map.py
    # fix = submission_array.clone()
    fix = submission_array.detach().cpu().numpy().copy()
    gt_array = gt_array.detach().cpu().numpy().copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return map(fix, gt_array)