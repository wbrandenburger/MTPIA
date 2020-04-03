# ===========================================================================
#   scores_regression.py ----------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.scores

import numpy as np
import pandas as pd

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_mean_absolute_error(truth, pred):
    """ Mean Absolute Error: 
    defined as average deviation of predicted class from true class """
    mean_absolute_error = np.mean(np.abs(np.subtract(truth, pred)))
    return mean_absolute_error

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_mean_squared_error(truth, pred):
    """ Mean Squared Error """
    mean_squared_error = np.mean(np.square(np.subtract(truth, pred)))
    return mean_squared_error

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_error_metric(truth, pred, metric="MAE"):
    """ Mean Absolute Error or mean squared error.""" 
    temp = np.subtract(truth, pred)
    if metric=="MAE":
        error = np.abs(temp)
    elif metric=="MSE":
        error = np.square(temp)
    return np.mean(metric)

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class RegressionScore(dl_multi.tools.scores.Scores):

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(
        self,
        number,
        categories=0,
        labels=None,
        label_spec=None,
        logger=None,
        decimals=3,
        **kwargs
    ):

        super(RegressionScore, self).__init__(number=number, logger=logger)

        self._categories = categories
        self._labels = labels if labels else range(self._categories) 
        self._label_spec = label_spec

        self._error = [None]*self._len
        self._error_per_class = [None]*self._len
        for index in range(self._len):
           self._error[index] = np.zeros((1, 2), dtype=np.float32)
           self._error_per_class[index] = np.zeros((self._categories, 3), dtype=np.float32)

        self._decimals=decimals

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update(self, truth, pred, label=None):
        self.logger("Compute error metric via 'dl-multi'.")
        self._error[self._index] = np.array([[get_mean_absolute_error(truth, pred), get_mean_squared_error(truth, pred)]])

        try:
            if label is None:
                return
        except NameError:
            return 

        for t_index, t_label in enumerate(self._labels):
            mask_label = label == t_label

            if mask_label.any():
                truth_masked = np.squeeze(truth[mask_label])
                pred_masked = np.squeeze(pred[mask_label])
                self._error_per_class[self._index][t_index, :] = np.array(
                    [
                        get_mean_absolute_error(truth_masked, pred_masked), 
                        get_mean_squared_error(truth_masked, pred_masked),
                        np.count_nonzero(mask_label)
                    ]
                )

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_error(self, index=None, current=False):
        index = self._index if current else index
        if index is not None:
            return self.get_error_util(self._error[index])
        
        return self.get_error_util(sum(self._error) / (self._index + 1))

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_error_per_class(self, index=None, current=False):
        index = self._index if current else index
        if index is not None:
            return self.get_error_util(self._error_per_class[index])
        
        return self.get_error_util(sum(self._error_per_class) / (self._index + 1))

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_error_util(self, error):    
        error[:,1] = np.sqrt(error[:,1], where=error[:,1]>0, out=error[:,1])
        if error.shape[1]>2:
            error[:,2] =  error[:,2] / np.sum(error[:,2])
            
        return np.where(error>0, error, -1.0)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_error_per_class_str(self, index=None, current=False):
        if self._label_spec:
            label_spec = self._label_spec
        else:
            label_spec = range(self._categories)

        return pd.DataFrame(
            self.get_error_per_class(index=index, current=current),
            columns=["MAE", "MSE", "Support"],
            index=self._label_spec
        ).round(decimals=self._decimals).to_string()

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_scores_str(self, index=None, current=False, verbose=False):
        index = self._index if current else index
        
        error_current = self.get_error(index=index, current=current)
        error = self.get_error()

        if index is not None:
           scores_str = "============ Stats regression step {}: {:.3f} (MAE), {:.3f}(MSE) / {:.3f} (MAE), {:.3f} (MSE) ".format(
                    index + 1,
                    error_current[0, 0],
                    error_current[0, 1],
                    error[0, 0],    
                    error[0, 1],
            )
        else:
            scores_str = "============ Stats regression: {:.3f} (MAE), {:.3f} (MSE)".format(
                error_current[0, 0],
                error_current[0, 1]
            )

            if verbose:
                scores_str = "{}\n\n{}".format(scores_str, self.get_error_per_class_str())

        return scores_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __repr__(self):
        return self.get_scores_str()


    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def write_log(self, log, write="w+", verbose=True):
        with open(log, write) as f:
            f.write(self.get_scores_str(verbose=verbose))    