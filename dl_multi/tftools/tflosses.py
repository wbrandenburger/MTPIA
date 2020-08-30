# ===========================================================================
#   tflosses.py -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from dl_multi.utils import imgtools

import os
import tensorflow as tf

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_accuracy(truth, pred, dtype=tf.float32):
    argmax = tf.to_float(tf.argmax(pred, axis=-1))
    truth = tf.to_float(truth)
    volume = tf.to_float(imgtools.get_volume(argmax.shape))

    accuracy = 1 - ( tf.count_nonzero((argmax - truth), dtype=dtype) / volume)
    
    return accuracy

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------

class Losses():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(
        self,
        obj,
        # number,
        # loss=None,

        task_weights=None,

        decimals=3,
        logger=None
    ):

        self._obj = obj if isinstance(obj, list) else [obj]

        self._tasks = len(self._obj)

        self._loss_handle = [None]*self._tasks
        for task in range(self._tasks):
            if self._obj[task] == "classification":
                    self._loss_handle[task] = tf.nn.sparse_softmax_cross_entropy_with_logits
            elif self._obj[task] == "regression":
                self._loss_handle[task] = tf.compat.v1.losses.mean_squared_error

        self._task_weights = task_weights if task_weights else [tf.to_float(1./self._tasks)]*self._tasks 

        self._loss_single_task = [tf.to_float(-1.)]*self._tasks
        self._single_task_supp = [tf.to_float(-1.)]*self._tasks

        self._logger = logger

        self._vis_divices = os.environ.get("CUDA_VISIBLE_DEVICES")

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def logger(self, log_str):
        if self._logger is not None:
            self._logger.info(log_str)
        return log_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update(self, truth, pred, weights=None, task_weights=None):
        # truth = truth if isinstance(truth, list) else [truth]   
        # pred = pred if isinstance(pred, list) else [pred]

        for task in range(self._tasks):
            if self._obj[task] == "classification":
                self.update_task_classification(task, truth, pred, weights)
            elif self._obj[task] == "regression":
                self.update_task_regression(task, truth, pred, weights)

        if not task_weights:
            task_weights = self._task_weights

        self._loss = tf.to_float(0.0)
        for task in range(self._tasks):
            self._loss += tf.to_float(self._loss_single_task[task] * task_weights[task])

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update_task_classification(self, task, truth, pred, weights):
        labels = tf.to_int32(tf.squeeze(tf.maximum(truth[task]-1, 0), axis=3))
        try:
            if len(weights):
                pass # weights = tf.to_float(tf.squeeze(tf.greater(truth[0], 0.))) without subtracting
        except TypeError:
                weights = tf.ones(labels.shape, tf.bool)
        self._loss_single_task[task] = tf.reduce_mean(
            tf.compat.v1.losses.compute_weighted_loss(losses=self._loss_handle[task](labels=labels, logits=pred[task]), weights = weights))

        self._single_task_supp[task] = get_accuracy(labels, pred[task])

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update_task_regression(self, task, truth, pred, weights):
        try:
            if len(weights):
                pass # weights = tf.to_float(tf.squeeze(tf.greater(truth[0], 0.)))
        except TypeError:
                weights = tf.ones(truth[task].shape, tf.bool)
        self._loss_single_task[task] = tf.compat.v1.losses.mean_squared_error(truth[task], pred[task], weights = weights) #tf.expand_dims(weights, axis=3) expand dims ?

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_loss(self, task=None):
        if task is not None:
            return self._loss_single_task[task]
        return self._loss

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_stats_str(self, index, stats):
        stats_str="==== cuda_device {}, step {} =>".format(self._vis_divices, index)
        stats_str="{} overall loss: {:.3f}".format(stats_str, stats["multi-task"])

        for task in range(self._tasks):
            stats_str="{} {}: {:.3f}".format(stats_str, self._obj[task], stats["single-task"][task])

            if stats["accuracy"][task] != -1:
                stats_str="{} ({:.3f})".format(stats_str, stats["accuracy"][task])

        return stats_str
    
    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_stats(self):
        return {"multi-task": self._loss, "single-task": self._loss_single_task, "accuracy": self._single_task_supp}
                

