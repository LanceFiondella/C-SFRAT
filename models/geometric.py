from core.model import Model


class Geometric(Model):
    name = "Geometric"
    converged = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calcHazard(self):
        pass



if __name__ == "__main__":
    g = Geometric()