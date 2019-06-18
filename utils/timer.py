'''
TIMER

Object used to measure time.

'''
import time


class Timer:
    def __init__(self):

        self.times = []
        self.total_timer = 0.
        self.timer = time.time()

    # start timer
    def start(self):
        self.timer = time.time()

    #stop timer
    def stop(self):
        self.total_timer += time.time() - self.timer
        self.timer = time.time()

    #get the current total times
    #reset timer
    #update time list
    def get_time(self):
        t = self.total_timer
        self.total_timer = 0
        self.times = [t] + self.times
        return t

    #get the current time list
    def list(self):
        return self.times
