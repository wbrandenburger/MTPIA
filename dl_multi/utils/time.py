# ===========================================================================
#   time.py -----------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import time

import numpy as np

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class MTime:

    def __init__(self, number=1, label="TIME"):
        self._len = number
        self._time_list = [0]*self._len
        self._label = "[{}]".format(label)
        
    def __iter__(self):
        self._time_list = [0]*self._len
        self._index = -1
        return self

    def __next__(self):
        if self._index < self._len-1:
            self._index += 1
            self.start()
            return self
        else:
            raise StopIteration

    def start(self):
        self._time_list[self._index] = -time.time()

    def stop(self):
        self._time_list[self._index] += time.time()

    def overall(self):
        return "{} time needed after {} of {}: {:02d}:{:02d}:{:02d}:{:02d}".format(self._label, self._index+1, self._len, *self.out(sum(self._time_list[0:self._index+1])))

    def stats(self):
        return "{} mean +/- std: {:01d}:{:02d}:{:02d}:{:02d} +/- {:02d}:{:02d}:{:02d}:{:02d} [d:h:m:s]".format(self._label,*self.out(np.mean(self._time_list[0:self._index+1])), *self.out(np.std(self._time_list[0:self._index+1])))

    def out(self, time):
        day = int(time // (24 * 3600))
        t_time = time % (24 * 3600)
        hour = int(time // 3600)
        t_time %= 3600
        minutes = int(time // 60)
        time %= 60
        seconds = int(time)
        return day, hour, minutes, seconds
