from typing import Tuple
from evogp.tree import Forest


class BaseSelection:
    def __call__(self, forest: Forest) -> Tuple[Forest, Forest]:
        raise NotImplementedError
