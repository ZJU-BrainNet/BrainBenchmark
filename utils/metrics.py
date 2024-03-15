# coding: UTF-8

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, precision_recall_curve



class BinaryClassMetrics:
    def __init__(self, args, pred, prob, true, ):
        pred, prob, true = pred.detach().numpy(), prob.detach().numpy(), true.detach().numpy()

        if args.dataset == 'Clinical' and args.run_mode == 'test':
            # remove label==2 in the Clinical inferring file
            pred = pred[true != 2]
            prob = prob[true != 2]
            true = true[true != 2]

        if np.sum(true) == 0 and np.sum(pred) == 0:
            self.special_good = True
        else:
            self.special_good = False

        self.tn = np.sum((pred == 0) & (true == 0))
        self.tp = np.sum((pred == 1) & (true == 1))
        self.fn = np.sum((pred == 0) & (true == 1))
        self.fp = np.sum((pred == 1) & (true == 0))
        # self.acc = np.sum(pred == true) / len(true)  # (tp+tn) / (tp+tn+fp+fn)
        # self.prec = self.tp / (self.tp + self.fp)
        # self.rec = self.tp / (self.tp + self.fn)
        # self.f_half = self.fbeta(self.prec, self.rec, beta=0.5)
        # self.f_one = self.fbeta(self.prec, self.rec, beta=1)
        # self.f_doub = self.fbeta(self.prec, self.rec, beta=2)
        self.acc = accuracy_score(true, pred)
        self.prec, self.rec, *_ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
        self.f_half = fbeta_score(true, pred, average='binary', beta=0.5)
        self.f_one  = fbeta_score(true, pred, average='binary', beta=1)
        self.f_doub = fbeta_score(true, pred, average='binary', beta=2)

        precision, recall, thresholds = precision_recall_curve(true, prob[:, 1])
        self.auc = auc(recall, precision)

    def get_confusion(self):
        return f"TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn} " if not self.special_good else "special_good"

    def get_metrics(self, one_line=False):
        if one_line:
            out = 'Acc:%.4f Prec:%.4f Rec:%.4f F1:%.4f F2:%.4f AUC:%.4f' \
                  % (self.acc, self.prec, self.rec, self.f_one, self.f_doub, self.auc)
        else:
            out = ''
            out += '-' * 15 + 'Metrics' + '-' * 15 + '\n'
            out += 'Accuracy:  ' + str(self.acc) + '\n'
            out += 'Precision: ' + str(self.prec) + '\n'
            out += 'Recall:    ' + str(self.rec) + '\n'
            # out += 'F0.5:      ' + str(self.f_half) + '\n'
            out += 'F1:        ' + str(self.f_one) + '\n'
            out += 'F2:        ' + str(self.f_doub) + '\n'
            out += 'AUC:       ' + str(self.auc) + '\n'
        return out if not self.special_good else "special_good"

    @staticmethod
    def fbeta(precision, recall, beta):
        return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)





class MultiClassMetrics:
    def __init__(self, basic_args, pred, prob, true, num_class, k=5,):
        pred, prob, true = pred.detach().numpy(), prob.detach().numpy(), true.detach().numpy()

        self.k = k
        self.acc = top_k_accuracy_score(true, prob, k=k, labels=np.arange(num_class))

        # self.prec_micro, self.rec_micro, *_ = precision_recall_fscore_support(true, pred, average='micro', zero_division=0)
        # self.f_half_micro = fbeta_score(true, pred, average='micro', beta=0.5)
        # self.f_one_micro  = fbeta_score(true, pred, average='micro', beta=1)
        # self.f_doub_micro = fbeta_score(true, pred, average='micro', beta=2)
        #
        # self.prec_macro, self.rec_macro, *_ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
        # self.f_half_macro = fbeta_score(true, pred, average='macro', beta=0.5)
        # self.f_one_macro  = fbeta_score(true, pred, average='macro', beta=1)
        # self.f_doub_macro = fbeta_score(true, pred, average='macro', beta=2)

        res = []
        for l in range(num_class):
            prec, recall, _, _ = precision_recall_fscore_support(true == l,
                                                                 pred == l,
                                                                 pos_label=True, average=None, zero_division=0)
            res.append([recall[0], recall[1]])  # 0:spec 1:sens
        res = np.array(res)
        self.spec_mean = np.mean(res[:, 0])
        self.sens_mean = np.mean(res[:, 1])

        self.f_one_macro  = fbeta_score(true, pred, average='macro', beta=1)
        self.kappa = cohen_kappa_score(true, pred)

        self.conf_matrix = ('\n' + str(confusion_matrix(true, pred))) if basic_args.print_matrix else ''

        self.f1_scores = f1_score(true, pred, average=None)

    def get_confusion(self):
        return self.conf_matrix

    def get_metrics(self, one_line=False):
        if one_line:
            out = f'Top {self.k} Acc:%.4f' % self.acc  + '\n'
            out += 'Sens:%.4f Spec:%.4f macroF1:%.4f Kappa:%.4f' % (self.sens_mean, self.spec_mean, self.f_one_macro, self.kappa) + '\n'
            out += 'F1 of each label: ' + np.array2string(self.f1_scores, precision=4,)
        else:
            out = ''
            out += '-' * 8 + f'Multiclass Metrics (Top {self.k} Acc)' + '-' * 8 + '\n'
            out += 'Accuracy:  ' + str(self.acc) + '\n'
            out += 'Sensitivity: ' + str(self.sens_mean) + '\n'
            out += 'Specificity: ' + str(self.spec_mean) + '\n'
            out += 'macro F1     ' + str(self.f_one_macro) + '\n'
            out += 'Kappa:       ' + str(self.kappa) + '\n'
        return out

    def draw_conf_matrix(self, config, annot=False, epoch=None):
        import seaborn as sb
        import pandas as pd
        # class_name = np.unique(np.concatenate([np.unique(self.pred), np.unique(self.true)]))
        class_name = np.unique(np.unique(self.true))

        cm_df = pd.DataFrame(self.conf_matrix,
                             index=class_name,
                             columns=class_name)

        f, (ax1, axcb) = plt.subplots(1, 2, figsize=(18, 18), gridspec_kw={'width_ratios': [1, 0.04]})
        cm_df = cm_df.apply(lambda x: (x / x.sum()), axis=1)
        g1 = sb.heatmap(cm_df, ax=ax1, cbar_ax=axcb, cmap="Blues", annot=annot, fmt='.2f', square=True)

        # g1 = sb.heatmap(cm_df, ax=ax1, cbar_ax=axcb, cmap="Blues", annot=annot, fmt='d', square=True)

        g1.set_title(f'exp{config.exp_id}_epoch{epoch}' if epoch is not None else \
                     f'exp{config.exp_id}')
        g1.set_ylabel('label')
        g1.set_xlabel('pred')
        plt.show()
        # plt.savefig(f'/home/zdz/confusion_tar1.eps', format='eps')


