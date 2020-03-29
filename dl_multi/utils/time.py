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

    def __init__(self, number=1, show=False):
        self._len = number
        self._time_list = [0]*self._len

        self._show=show
        
    def __iter__(self):
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

    def stop(self, show=False):
        self._time_list[self._index] += time.time()
        if show or self._show:
            print(self)

    @property
    def overall(self):
        return sum(self._time_list)

    def __repr__(self):
        t_time = self.overall
        day = t_time // (24 * 3600)
        t_time = t_time % (24 * 3600)
        hour = t_time // 3600
        t_time %= 3600
        minutes = t_time // 60
        t_time %= 60
        seconds = t_time
        return "Time needed in d:h:m:s-> {:02d}:{:02d}:{:02d}:{:02d}".format(int(day), int(hour), int(minutes), int(seconds))

    def stats(self):
        print('Mean and standard deviation in seconds {:.3f} {:.3f}'.format(np.mean(self._time_list), np.std(self._time_list)))