# ===========================================================================
#   scores_classification.py ------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.metrics.scores

import numpy as np
import pandas as pd

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_overall_accuracy(confusion_matrix):
    """Compute the overall accuracy.
    """
    return np.trace(confusion_matrix)/confusion_matrix.sum()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_support_per_class(confusion_matrix):
    """Compute the number of elements in confusion matrix
    """
    sums = np.sum(confusion_matrix, axis=1)
    support = np.sum(sums)
    support_per_class = sums/support
    return 1.0, support_per_class

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_pfa_per_class(confusion_matrix):
    """Compute the probability of false alarms.
    """
    sums = np.sum(confusion_matrix, axis=0) 
    mask = (sums>0)
    sums[sums==0] = 1
    pfa_per_class = (confusion_matrix.sum(axis=0)-np.diag(confusion_matrix)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    # average_pfa = pfa_per_class[mask].mean()
    average_pfa = np.sum(np.multiply(pfa_per_class[mask], get_support_per_class(confusion_matrix)[1][mask]))
    return average_pfa, pfa_per_class

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_accuracy_per_class(confusion_matrix):
    """Compute the accuracy per class and average
        puts -1 for invalid values (division per 0)
        returns average accuracy, accuracy per class
    """
    # equvalent to for class i to 
    # number or true positive of class i (data[target==i]==i).sum()/ number of elements of i (target==i).sum()
    sums = np.sum(confusion_matrix, axis=1) 
    mask = (sums>0)
    sums[sums==0] = 1
    accuracy_per_class = np.diag(confusion_matrix) / sums #sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    # average_accuracy = accuracy_per_class[mask].mean()
    average_accuracy = np.sum(np.multiply(accuracy_per_class[mask], get_support_per_class(confusion_matrix)[1][mask]))
    return average_accuracy, accuracy_per_class    

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_iou_per_class(confusion_matrix):
    """Compute the iou per class and average iou
        Puts -1 for invalid values
        returns average iou, iou per class
    """
    
    sums = (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    mask  = (sums>0)
    sums[sums==0] = 1
    iou_per_class = np.diag(confusion_matrix) / sums
    iou_per_class[np.logical_not(mask)] = -1
    # average_iou = iou_per_class[mask].mean()
    average_iou = np.sum(np.multiply(iou_per_class[mask], get_support_per_class(confusion_matrix)[1][mask]))
    return average_iou, iou_per_class

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_f1score_per_class(confusion_matrix):
    """Compute f1 scores per class and mean f1.
        puts -1 for invalid classes
        returns average f1 score, f1 score per class
    """
    # defined as 2 * recall * prec / recall + prec
    sums = (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0))
    mask  = (sums>0)
    sums[sums==0] = 1
    f1score_per_class = 2 * np.diag(confusion_matrix) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    # average_f1_score =  f1score_per_class[mask].mean()
    average_f1_score = np.sum(np.multiply(f1score_per_class[mask], get_support_per_class(confusion_matrix)[1][mask]))
    return average_f1_score, f1score_per_class

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class ClassificationScore(dl_multi.metrics.scores.Scores):

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(
        self,
        number,
        categories=0,
        labels=None,
        label_spec=None,
        sklearn=True,
        logger=None,
        decimals=3,
        **kwargs
    ):

        super(ClassificationScore, self).__init__(number=number, logger=logger)
        self._categories = categories
        self._labels = labels if labels else range(self._categories) 
        self._label_spec = label_spec
        self._confusion_matrix = [None]*self._len
        for index in range(self._len):
            self._confusion_matrix[index] = np.zeros([self._categories]*2, dtype=np.uint32)

        self._sklearn=sklearn
        self._decimals=decimals

        self._temp_confusion_matrix = np.zeros(0)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def update(self, truth, pred, **kwargs):
        self._temp_confusion_matrix = np.zeros(0)

        if self._sklearn:
            try: 
                from sklearn.metrics import confusion_matrix
                self.logger("Compute confusion matrix via 'sklearn' with labels {}.".format(self._labels))
                self._confusion_matrix[self._index] = confusion_matrix(truth.ravel(), pred.ravel(), labels=self._labels)
                return
                # tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
            except ImportError:
                pass

        self.logger("Compute confusion matrix via 'dl-multi' with labels {}.".format(self._labels))
        for t_label in self._labels:
            truth_label = truth == t_label
            for p_label in self._labels:
                pred_label = pred == p_label
                self._confusion_matrix[self._index][t_label, p_label] = np.count_nonzero(np.logical_and(truth_label == pred_label, truth_label))

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_confusion_matrix(self, index=None, current=False):
        index = self._index if current else index
        if index is not None:
            return self._confusion_matrix[index]
        
        if not len(self._temp_confusion_matrix):
            self._temp_confusion_matrix = sum(self._confusion_matrix)
        return self._temp_confusion_matrix
    
    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_overall(self, index=None, current=False):
        confusion_matrix = self.get_confusion_matrix(index=index, current=current)
        return np.round(get_overall_accuracy(confusion_matrix), decimals=self._decimals)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_scores_per_class_str(self, index=None, current=False):
        confusion_matrix = self.get_confusion_matrix(index=index, current=current)

        acc, acc_per_class = get_accuracy_per_class(confusion_matrix)
        iou, iou_per_class = get_iou_per_class(confusion_matrix)
        f1_score, f1score_per_class = get_f1score_per_class(confusion_matrix)
        pfa, pfa_per_class = get_pfa_per_class(confusion_matrix)
        support, support_per_class = get_support_per_class(confusion_matrix)

        if self._label_spec:
            label_spec = self._label_spec.copy()
            label_spec.append("Weighted mean")
        else:
            label_spec = range(self._categories+1)

        return pd.DataFrame(
            np.concatenate(
                (
                    np.stack((acc_per_class, iou_per_class, f1score_per_class, pfa_per_class, support_per_class)).T,
                    np.array([[acc, iou, f1_score, pfa, support]])
                )
            ),
            columns=["Accuracy", "IoU", "F1 Score", "P(FA)", "Support"],
            index=label_spec
        ).round(decimals=self._decimals).to_string()

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def get_scores_str(self, index=None, current=False, verbose=False, **kwargs):
        index = self._index if current else index
        
        if index is None:
            scores_str = "============ Overall Stats - Classification: {}".format(
                self.get_overall(index=index, current=current)
            )
        else:
            scores_str = "============ Object Stats - Classification: {}".format(
                    self.get_overall(index=index, current=current)
            )

            if verbose:
                scores_str = "{}\n\n{}".format(scores_str, self.get_scores_per_class_str(index=index, current=current))

        return scores_str

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __repr__(self):
        return self.get_scores_str()

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def write_log(self, log, write="w+", **kwargs):
        with open(log, write) as f:
            f.write(self.get_scores_str(**kwargs))              