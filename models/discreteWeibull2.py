from core.model import Model


class DiscreteWeibull2(Model):
    name = "Discrete Weibull (Order 2)"
    converged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        pass

    def calcHazard(self):
        pass



if __name__ == "__main__":
    dw = DiscreteWeibull2()