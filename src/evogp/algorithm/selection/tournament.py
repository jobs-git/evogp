from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection
from functools import partial


class TournamentSelection(BaseSelection):

    def __init__(
        self,
        tournament_size: int,
        best_probability: float = 1,
        replace: bool = True,
        survivor_rate: float = 0.5,
        elite_rate: float = 0,
        survivor_cnt: Optional[int] = None,
        elite_cnt: Optional[int] = None,
    ):
        super().__init__()
        assert 0 <= survivor_rate <= 1, "survival_rate should be in [0, 1]"
        assert 0 <= elite_rate <= 1, "elite_rate should be in [0, 1]"
        self.t_size = tournament_size
        self.best_p = best_probability
        self.replace = replace
        self.survivor_rate = survivor_rate
        self.survivor_cnt = survivor_cnt
        self.elite_rate = elite_rate
        self.elite_cnt = elite_cnt

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        @partial(torch.vmap, randomness="different")
        def traverse_once(p):
            return torch.multinomial(
                p, n_tournament * self.t_size, replacement=self.replace
            ).to(torch.int32)

        @torch.vmap
        def t_selection_without_p(contenders):
            contender_fitness = fitness[contenders]
            best_idx = torch.argmax(contender_fitness)[None]
            return contenders[best_idx]

        @partial(torch.vmap, randomness="different")
        def t_selection_with_p(contenders):
            contender_fitness = fitness[contenders]
            idx_rank = torch.argsort(
                contender_fitness, descending=True
            )  # the index of individual from high to low
            random = torch.rand(1).cuda()
            best_p = torch.tensor(self.best_p).cuda()
            nth_choosed = (torch.log(random) / torch.log(1 - best_p)).to(torch.int32)
            nth_choosed = torch.where(
                nth_choosed >= self.t_size, torch.tensor(0), nth_choosed
            )
            return contenders[idx_rank[nth_choosed]]

        # preprocess
        if self.survivor_cnt is not None:
            survivor_cnt = self.survivor_cnt
        else:
            survivor_cnt = int(forest.pop_size * self.survivor_rate)

        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        else:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # survivor selection
        total_size = fitness.size(0)
        n_tournament = int(total_size / self.t_size)
        k_times = int((survivor_cnt - 1) / n_tournament) + 1
        p = torch.ones((k_times, total_size)).cuda()
        contenders = traverse_once(p).reshape(-1, self.t_size)[:survivor_cnt]

        if self.t_size > 1000:
            survivor_indices = t_selection_without_p(contenders).reshape(-1)
        else:
            survivor_indices = t_selection_with_p(contenders).reshape(-1)

        # elite selection
        if elite_cnt == 0:
            elite_indices = torch.tensor([], device="cuda", dtype=torch.int64)
        else:
            sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
            elite_indices = sorted_indices[:elite_cnt]

        return elite_indices, survivor_indices
