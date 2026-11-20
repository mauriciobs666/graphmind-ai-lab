class P:
    def step(self):
        return 1

    def run(self):
        return self.step()


class Q(P):
    def step(self):
        return 2
