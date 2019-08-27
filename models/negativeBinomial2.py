from core.model import Model


class NegativeBinomial2(Model):
    name = "Negative Binomial (Order 2)"
    converged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calcHazard(self):
        pass



if __name__ == "__main__":
    nb = NegativeBinomial2()