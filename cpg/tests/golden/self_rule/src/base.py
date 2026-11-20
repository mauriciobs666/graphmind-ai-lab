class Base:
    def step(self):
        return 1

    def run(self):
        return self.step()

    def other(self):
        return self.only_here()

    def only_here(self):
        return 5
