
import time


class Timer:
    def __init__(self):

        self.times = []
        self.total_timer = 0.
        self.timer = time.time()

    def start(self):
        self.timer = time.time()

    def stop(self):
        self.total_timer += time.time() - self.timer
        self.timer = time.time()

    def get_time(self):
        t = self.total_timer
        self.total_timer = 0
        self.times = [t] + self.times
        return t

    def list(self):
        return self.times
