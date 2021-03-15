from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt


class Metric:
    """Computes:
  
    1. Average Precision Score
    2. Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    3. Receiver operating characteristic (ROC) for binary classification task

    Args:
    y_true- np array containing ground truth values
    y_score- np array of predictions after activation has been applied

    """

    def __init__(self, y_true, y_score):
        self.average_precision_score_fn(y_true, y_score)
        self.roc_auc_score_fn(y_true, y_score)
        self.roc_curve_fn(y_true, y_score)

    def average_precision_score_fn(self, y_true, y_score):
        print("\nAVERAGE PRECISION SCORE")
        print(average_precision_score(y_true, y_score))

    def roc_auc_score_fn(self, y_true, y_score):
        print("\nROC AUC SCORE")
        print(roc_auc_score(y_true, y_score))

    def roc_curve_fn(self, y_true, y_score):
        print("\nROC CURVE")
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        plt.plot(fpr[1], tpr[1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
