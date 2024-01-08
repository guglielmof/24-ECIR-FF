from time import time


class Timer:

    def __init__(self):
        self.start = None
        self.started = False

    def cp(self, message=None):
        if self.started:
            message = f"done in {time() - self.start:.2f}"

        print(message)
        self.started = not self.started
        self.start = time()
