import torch
from . import BaseProblem
from evogp.tree import Forest

from sklearn.datasets import load_diabetes


class Transformation(BaseProblem):
    def __init__(self, dataset: str):
        self.datapoints, self.labels = self.generate_data(dataset)

    def generate_data(self, dataset: str):
        if dataset == "diabetes":
            X, y = load_diabetes(return_X_y=True)
        else:
            raise ValueError("Invalid dataset")
        inputs = torch.tensor(X, dtype=torch.float32, device="cuda")
        labels = torch.tensor(y, dtype=torch.float32, device="cuda")
        return inputs, labels

    def evaluate(self, forest: Forest):
        outputs = forest.batch_forward(self.datapoints).squeeze()
        outputs_demean = outputs - torch.mean(outputs)
        labels_demean = self.labels - torch.mean(self.labels)
        correlation = torch.sum(outputs_demean * labels_demean, dim=1) / torch.sqrt(
            torch.sum(outputs_demean**2, dim=1) * torch.sum(labels_demean**2)
        )
        return torch.abs(correlation)

    def new_feature(self, forest: Forest, n_best, n_features):
        def worthy_correlation(correlations, best, n):
            selected = torch.ones(best.shape[0], dtype=torch.bool).cuda()
            while torch.sum(selected) > n:
                most_correlated_idx = torch.unravel_index(
                    torch.argmax(correlations), correlations.shape
                )
                worst = max(most_correlated_idx)
                selected[worst] = False
                correlations[worst, :] = 0
                correlations[:, worst] = 0

            return selected

        fitness = self.evaluate(forest)
        best = fitness.argsort(descending=True)[:n_best]
        forward = forest[best].batch_forward(self.datapoints).squeeze()
        correlations = torch.abs(torch.corrcoef(forward))
        mask = torch.eye(
            correlations.size(0), device=correlations.device, dtype=torch.bool
        )
        correlations.masked_fill_(mask, 0)
        worthy = worthy_correlation(correlations, best, n_features)
        gp_features = forest[best][worthy].batch_forward(self.datapoints).squeeze().T
        return gp_features
