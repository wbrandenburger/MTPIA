# ===========================================================================
#   metrics.py -----------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.utils import imgtools
import dl_multi.metrics.scores_classification
import dl_multi.metrics.scores_regression

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Metrics():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(
        self,
        obj,

        number,
        # loss=None,

        categories=None,
        labels=None,
        label_spec=None,

        decimals=3,
        logger=None,
        sklearn=True
    ):
        self._obj = obj if isinstance(obj, list) else [obj]

        self._len = number
        self._tasks = len(self._obj)

        self._scores = list()
        for task in range(self._tasks):
            if self._obj[task] == "classification":
                self._scores.append(dl_multi.metrics.scores_classification.ClassificationScore(
                    number,
                    categories=categories,
                    labels=labels,
                    label_spec=label_spec,
                    sklearn=sklearn,
                    logger=logger,
                    decimals=decimals
                )
            )
            # elif obj[task] == "ordinal_regression":
            #     self._scores.append(dl_multi.metrics.scores.OrdinalRegressionScores())
            elif self._obj[task] == "regression":
                self._scores.append(dl_multi.metrics.scores_regression.RegressionScore(
                    number,
                    categories=categories,
                    labels=labels,
                    label_spec=label_spec,
                    logger=logger,
                    decimals=decimals
                )
            )

        self._logger = logger

        self._index = -1
        
    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __len__(self):
        return self._len

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __iter__(self):
        self._index = -1
        for index in range(len(self._scores)):
            iter(self._scores[index])

        return self

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __next__(self):
        if self._index < self._len-1:
            self._index += 1
            for index in range(len(self._scores)):
                next(self._scores[index])
            return self
        else:
            raise StopIteration

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def logger(self, log_str):
        if self._logger is not None:
            self._logger.debug(log_str)
        return log_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update(self, truth, pred, label=None):
        truth = truth if isinstance(truth, list) else [truth]
        pred = pred if isinstance(pred, list) else [pred] 
        for task in range(len(self._scores)):
            self._scores[task].update(truth[task], pred[task], label=label)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def print_current_stats(self):
        metric_str="================ Stats tasks:"
        for task in range(len(self._scores)):
            metric_str="{}\n\n{}".format(metric_str, self._scores[task].get_scores_str(current=True))
        return metric_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __repr__(self):
        metric_str="================ Stats tasks:"
        for task in range(len(self._scores)):
            metric_str="{}\n\n{}".format(metric_str, self._scores[task].get_scores_str(verbose=True))
        return metric_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def write_log(self, log, write="w+", **kwargs):
        if not isinstance(log, list):
            self.logger("[SAVE] '{}'".format(log))
            for task in range(len(self._scores)):
                self._scores[task].write_log(log, write=write, **kwargs)

                write = "a+"
                with open(log, write) as f:
                    f.write("\n\n")
        else:
            for task in range(len(self._scores)):
                self.logger("[SAVE] '{}'".format(log[task]))
                self._scores[task].write_log(log[task], write=write, **kwargs)