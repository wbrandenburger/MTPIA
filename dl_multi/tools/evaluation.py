# ===========================================================================
#   evaluation.py -----------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.tools.imgtools
import dl_multi.tools.welford

import sklearn
import numpy as np
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class EvalCat():

    def __init__(self, label, index, log=None):
        self._log=log
        self._eval_cat = dl_multi.tools.evaluation.CatScores(list(label.values()), index=index, log=self._log)
        self._eval_map = dl_multi.tools.evaluation.ConfusionMap(list(label.values()), index=index, log=self._log) 
    
    def update(self, pred, truth):
        self._eval_cat.update(pred, truth)
        self._eval_map.update(pred, truth)

    def write_log(self, log=None, scores=None):
        log = log if log else self._log 

        self._eval_cat.write_log(log=log)
        self._eval_map.write_log(log=log)

    def imsave(self, path):
        self._eval_map.imsave(path)

    def plot(self):
        self._eval_map.plot()

    def __repr__(self):
        return "{}\n\n{}".format(self._eval_cat, self._eval_map)

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class EvalReg():

    def __init__(self, log=None):

        self._scores = [dl_multi.tools.welford.Welford(), dl_multi.tools.welford.Welford()]

        self._log = log
        if os.path.exists(self._log):
            os.remove(self._log)

    def update(self, pred, truth):
        scores = [dl_multi.tools.welford.Welford(), dl_multi.tools.welford.Welford()]
        pred = dl_multi.tools.imgtools.project_data_to_img(pred, dtype=np.float32)
        truth = dl_multi.tools.imgtools.project_data_to_img(truth, dtype=np.float32)
        self._scores[0].update(np.mean(np.absolute(pred-truth)))
        self._scores[1].update(np.std(pred-truth))
    
    def __repr__(self):
        return "{} {}".format(self._scores[0],self._scores[1])

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class ConfusionMap():

    # https://www.sciencedirect.com/topics/engineering/confusion-matrix

    def __init__(self, labels, index=None, log=None):
        self._labels = labels
        self._data_frame_index = index
        if not index:
            self._data_frame_index = range(len(self.labels))

        self._confusion_map = np.zeros((len(self._labels), len(self._labels)), dtype=int)
                
        self._log = log
        if os.path.exists(self._log):
            os.remove(self._log)
      
    def update(self, pred, truth):
        confusion_map = np.zeros((len(self._labels), len(self._labels)), dtype=int)
        for r in self._labels:
            truth_r = truth == r
            for c in self._labels:
                pred_c = pred == c
                confusion_map[r, c] = np.count_nonzero(
                    np.logical_and(truth_r == pred_c, truth_r)
                )
            
        # if self._log: self.write_log(confusion_map=confusion_map)
        self._confusion_map += confusion_map

    @property
    def confusion_map(self):
        #             ground  vegetation  buildings   water  bridge  clutter
        # ground      212283      588561     385686  400471       0   188856
        # vegetation   57340      392315      29281  314961       0    25680
        # buildings    49565       95952     111132   65534       0    64904
        # water          472       10878        217   70172       0      270
        # bridge           7           7         23       5       0        4
        # clutter       9901       24794      19063   17592       0     9802
        
        # label 'row' is true but label 'column' was (mis)classified
        return self.get_data_frame(confusion_map=self._confusion_map)

    def get_confusion_map(self, confusion_map=None):
        num = (confusion_map if hasattr(confusion_map, "len") else self._confusion_map).astype(float)
        denom = np.sum(num, axis=1).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)

    def get_data_frame(self, confusion_map=None):
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)
        data = pd.DataFrame(
            confusion_map,
            columns=self._data_frame_index,
            index=self._data_frame_index)
        return data
    
    def imsave(self, path, confusion_map=None):
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)                
        sn.heatmap(self.get_data_frame(confusion_map=confusion_map), annot=True)
        plt.savefig(path)

    def plot(self, confusion_map=None):
        # confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)
        sn.heatmap(self.get_data_frame(confusion_map=confusion_map), annot=True)

    def to_string(self, confusion_map=None):
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)
        return self.get_data_frame(confusion_map=confusion_map.round(3)).to_string()

    def write(self, confusion_map=None):
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)
        return self.to_string(confusion_map=confusion_map)

    def write_log(self, log=None, confusion_map=None):
        confusion_map = self.get_confusion_map(confusion_map=confusion_map)

        log = log if log else self._log 
        with open(log, 'a+') as f:
            f.write("\nConfusion Map\n{}\n\n".format(self.to_string(confusion_map=confusion_map)))

    def __repr__(self):
        return self.to_string()


#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class CatScores:

    def __init__(self, labels, index=None, log=None):
        self._labels = labels
        self._data_frame_index = index

        if not index:
            self._data_frame_index = range(len(self.labels))

        # tp, fp, tn, fn (p=tp+fp, n=tn,fn)
        self._scores = np.zeros((len(self._labels), 4), dtype=float)

        self._log = log 
        if os.path.exists(self._log):
            os.remove(self._log)

    def update(self, pred, truth):
        scores = np.zeros((len(self._labels), 4), dtype=float)
        # ignore pixel with value 6
        # acc = (np.count_nonzero( img_out == label) - np.sum(img_out==6))/(label.shape[0]*label.shape[1] - np.sum(label==6))
        acc = np.count_nonzero(pred==truth)/(pred.shape[0]*pred.shape[1])
        for l in self._labels:
            l_pred = pred == l
            l_truth = truth == l

            l_truth_and_pred = l_truth==l_pred
            l_truth_not_pred = l_truth!=l_pred
            l_truth_inverted = np.invert(l_truth)
            scores[l,0] += np.count_nonzero(
                np.logical_and(l_truth_and_pred, l_truth)) # tp
            scores[l,1] += np.count_nonzero(
                np.logical_and(l_truth_not_pred, l_truth)) # fp
            scores[l,2] += np.count_nonzero(
                np.logical_and(l_truth_and_pred , l_truth_inverted))  # tn
            scores[l,3] += np.count_nonzero(
                np.logical_and(l_truth_not_pred, l_truth_inverted))  # fn
  
        # if self._log: self.write_log(scores=scores)
        self._scores += scores

    def get_overall_acc(self, scores=None):
        scores = self.get_scores(scores=scores)
        # overall accuracy: (tp + tn) / (tp + tn + fp + fn)
        a = scores[:,0]
        b = self.get_support(scores=scores)
        return np.sum(a) / np.sum(b)

    @property
    def overall_acc(self):
        return self.get_overall_acc(scores=self._scores)

    def get_precision(self, scores=None):
        # positive predicitve value: tp / (tn + fp)
        # Precision is the estimated probability that a randomly selected retrieved document is relevant
        scores = self.get_scores(scores=scores)
        num = (scores[:,0]).astype(float)
        denom = (scores[:,1] + scores[:,2]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)

    @property
    def precision(self):
        return self.get_precision(scores=self._scores)

    def get_recall(self, scores=None):
        # true positive rate: tp / (tn + fp)
        # Recall is the estimated probability that a randomly selected relevant document is retrieved in a search
        scores = self.get_scores(scores=scores)
        num = (scores[:,0]).astype(float)
        denom = (scores[:,2] + scores[:,3]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    
    @property
    def recall(self):
        return self.get_recall(scores=self._scores)

    def get_selectivity(self, scores=None):
        # true negative rate: tn / (tn + fp)
        scores = self.get_scores(scores=scores)
        num = (scores[:,1]).astype(float)
        denom = (scores[:,2] + scores[:,3]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)

    @property
    def selectivity(self):
        return self.get_selectivity(scores=self._scores)

    def get_acc(self, scores=None):
        # accuracy: (tp + tn) / (tp + tn + fp + fn)
        scores = self.get_scores(scores=scores)
        num = (scores[:,0] + scores[:,2]).astype(float)
        denom = (scores[:,0] + scores[:,2] + scores[:,1] + scores[:,3]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=num) 

    @property
    def acc(self):
        return self.get_acc(scores=self._scores)

    def get_bacc(self, scores=None):
        # balanced accuracy: (tpr + tnr) / 2
        scores = self.get_scores(scores=scores)
        return (self.get_precision(scores=scores) + self.get_selectivity(scores=scores))/2

    @property
    def bacc(self):
        return self.get_bacc(scores=self._scores)

    def get_f1_score(self, scores=None):
        # f1_score: 2tp / (2tp + fp + fn)
        scores = self.get_scores(scores=scores)
        num = (2*scores[:,0]).astype(float)
        denom = (2*scores[:,0] + scores[:,1] + scores[:,3]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)

    @property 
    def f1_score(self):
        return self.get_f1_score(scores=self._scores)

    def get_iou(self, scores=None):
        # intersect of union: tp / (tp + fp + fn)
        scores = self.get_scores(scores=scores)
        num = (scores[:,0]).astype(float)
        denom = (scores[:,0] + scores[:,1] + scores[:,3]).astype(float)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)

    @property 
    def iou(self):
        return self.get_iou(scores=self._scores)

    def get_support(self, scores=None):
        # support: tp + fp
        scores = self.get_scores(scores=scores)
        return scores[:,0] + scores[:,1]

    @property 
    def support(self):
        return self.get_support(scores=self._scores)

    def get_scores(self, scores=None):
        return scores if hasattr(scores,"len") else self._scores

    @property
    def scores(self):
        return self.get_data_frame(scores=self._scores)

    def get_data_frame(self, scores=None):
        scores = self.get_scores(scores=scores)
        data = pd.DataFrame(
            np.stack((self.get_acc(scores=scores), self.get_bacc(scores=scores), self.get_precision(scores=scores), self.get_recall(scores=scores), self.get_f1_score(scores=scores), self.get_iou(scores=scores), self.get_support(scores=scores))).T,
            columns=["Accuracy", "Balanced Acc", "Precision", "Recall", "F1 Score", "IoU", "Support"],
            index=self._data_frame_index)
        return data

    def to_string(self, scores=None):
        scores = self.get_scores(scores=scores)
        return self.get_data_frame(scores=scores.round(3)).to_string()

    def write(self, scores=None):
        scores = self.get_scores(scores=scores)
        return self.to_string(scores=scores)

    def write_log(self, log=None, scores=None):
        scores = self.get_scores(scores=scores)
        log = log if log else self._log 
        with open(log, 'a+') as f:
            f.write("\nClassification report with overall accuracy {}\n{}\n\n".format(self.overall_acc, self.to_string(scores=scores)))

    def __repr__(self):
        return self.to_string()

# #   class -------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# class CatScores:

#     def __init__(self, labels, index=None, log=None):
#         self._labels = labels
#         self._data_frame_index = index

#         if not index:
#             self._data_frame_index = range(len(self.labels))

#         # tp, fp, tn, fn (p=tp+fp, n=tn,fn)
#         self._scores = np.zeros((len(self._labels), 4), dtype=float)

#         self._log = log 
#         if os.path.exists(self._log):
#             os.remove(self._log)

#     def update(self, pred, truth):
#         scores = np.zeros((len(self._labels), 4), dtype=float)
#         # ignore pixel with value 6
#         # acc = (np.count_nonzero( img_out == label) - np.sum(img_out==6))/(label.shape[0]*label.shape[1] - np.sum(label==6))
#         acc = np.count_nonzero(pred==truth)/(pred.shape[0]*pred.shape[1])
#         for l in self._labels:
#             l_pred = pred == l
#             l_truth = truth == l

#             l_truth_and_pred = l_truth==l_pred
#             l_truth_not_pred = l_truth!=l_pred
#             l_truth_inverted = np.invert(l_truth)
#             scores[l,0] += np.count_nonzero(
#                 np.logical_and(l_truth_and_pred, l_truth)) # tp
#             scores[l,1] += np.count_nonzero(
#                 np.logical_and(l_truth_not_pred, l_truth)) # fp
#             scores[l,2] += np.count_nonzero(
#                 np.logical_and(l_truth_and_pred , l_truth_inverted))  # tn
#             scores[l,3] += np.count_nonzero(
#                 np.logical_and(l_truth_not_pred, l_truth_inverted))  # fn
  