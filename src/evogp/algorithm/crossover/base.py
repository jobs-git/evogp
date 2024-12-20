from ...tree import Forest

class BaseCrossover:
    def __call__(self, forest: Forest):
        raise NotImplementedError
