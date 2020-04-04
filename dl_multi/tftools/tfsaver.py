# ===========================================================================
#   tfsaver.py --------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Saver():
    
    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(
        self, 
        saver,
        num_epochs, 
        iteration=1000, 
        steps=10000,
        logger=None
    ):
        self._saver = saver
        self._len = num_epochs

        self._iteration = iteration
        self._steps = steps
        # self.set_steps(steps)

        self._logger = logger

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __len__(self):
        return self._len

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __iter__(self):
        self._index = -1
        return self

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __next__(self):
        if self._index < self._len:
            self._index += 1
            return self
        else:
            raise StopIteration

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def logger(self, log_str):
        if self._logger is not None:
            self._logger.debug(log_str)
        return log_str

    # #   method --------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # def set_steps(self, steps):
    #     self._steps=steps
    #     if not isinstance(steps, list):
    #         self._steps = range(0, self._len, steps)
    #     if not self._len in self._steps:
    #         self.steps_append(self._len)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def save(self, session, checkpoint, step=True):
        
        global_step=self._index

        if step:
            save_ckpt = False
            if self._index % self._iteration == 0:
                save_ckpt = True
                global_step = None

            if self._index % self._steps == 0:
                save_ckpt = True
                global_step=self._index

        if save_ckpt:
            session_file = self._saver.save(session, checkpoint, global_step=global_step)
            self.logger("[SAVE] Model saved in file: {}".format(
                    session_file))