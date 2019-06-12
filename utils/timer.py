
import time


class Timer:
    def __init__(self):

        self.times = []
        self.total_timer = 0.
        self.timer = time.time()

    def start_time():
        global timer
        timer = time.time()

    def stop_time():
        global total_timer, timer
        total_timer += time.time() - timer
        timer = time.time()

    def get_time():
        global total_timer
        t = total_timer
        total_timer = 0
        return t
