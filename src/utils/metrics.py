import torch
from torchmetrics import Metric


class PRFMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("t_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, vid_p, vid_r) -> None:
        self.t_precision += vid_p
        self.t_recall += vid_r
        self.t_f1 += 2 * (vid_p * vid_r) / (vid_p + vid_r) if vid_p + vid_r else 0.0
        self.n += 1

    def compute(self):
        avg_p = self.t_precision * 100 / self.n
        avg_r = self.t_recall * 100 / self.n
        avg_f1 = self.t_f1 * 100 / self.n
        return {"precision": avg_p, "recall": avg_r, "f1": avg_f1}
