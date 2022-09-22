# -*- coding: utf-8 -*-
import os

import pandas as pd
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, accuracy_score, \
    precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, plot_roc_curve

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# def cal_threshold_performances(clf, X, Y, threshold):
#     prob = clf.predict_proba(X)
#     prob = prob[:, 1]
#     res = [1 if i >= threshold else 0 for i in prob]
#
#     auc_score = roc_auc_score(Y, res)
#
#     a = confusion_matrix(Y, res)
#     tp = a[1, 1]
#     tn = a[0, 0]
#     fn = a[1, 0]
#     fp = a[0, 1]
#     sensitity = tp / (tp + fn)   # 1 - false positive rate
#     specifity = tn / (tn + fp)   # true positive rate
#
#     return auc_score, sensitity, specifity

# 求阈值的函数
def threshold_prob(clf, X, Y, sensitivity_threshold=0.75):

    # sensitity = tp / (tp + fn)   # 1 - false positive rate
    # specifity = tn / (tn + fp)   # true positive rate

    y_one_hot = label_binarize(Y, classes=np.arange(2))  # 装换成类似二进制的编码
    print(y_one_hot.shape)
    print(clf.predict_proba(X).shape)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_one_hot.ravel(), clf.predict_proba(X)[:, 1].ravel()
    )

    best_sensitity = None
    best_specifity = None
    best_threshold = None

    for fpr, tpr, thres in zip(false_positive_rate, true_positive_rate, thresholds):
        sensitity = 1 - fpr
        specifity = tpr

        if sensitity >= sensitivity_threshold:
            if not best_specifity:
                best_sensitity = sensitity
                best_specifity = specifity
                best_threshold = thres
            else:
                if specifity > best_specifity:
                    best_sensitity = sensitity
                    best_specifity = specifity
                    best_threshold = thres

    return best_sensitity, best_specifity, best_threshold


def threshold_prob_via_jordan(clf, X, Y, ):
    # 通过约当指数求最佳阈值

    # sensitity = tp / (tp + fn)   # 1 - false positive rate
    # specifity = tn / (tn + fp)   # true positive rate

    y_one_hot = label_binarize(Y, classes=np.arange(2))  # 装换成类似二进制的编码
    print(y_one_hot.shape)
    print(clf.predict_proba(X).shape)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_one_hot.ravel(), clf.predict_proba(X)[:, 1].ravel()
    )

    best_jordan_index = -1.0
    best_sensitity = None
    best_specifity = None
    best_threshold = None

    for fpr, tpr, thres in zip(false_positive_rate, true_positive_rate, thresholds):
        sensitity = 1 - fpr
        specifity = tpr
        jordan_ = specifity + sensitity

        if jordan_ >= best_jordan_index:
            best_jordan_index = jordan_
            best_sensitity = sensitity
            best_specifity = specifity
            best_threshold = thres

    return best_sensitity, best_specifity, best_threshold


def train_cv_with_thres(samples,
                        clf,
                        features,
                        num_folds=5,
                        random_state=101,
                        sensitivity_threshold=0.75,
                        save_dir=None,
                        ):
    # pos_samples = samples[samples['label'] == 1]
    # neg_samples = samples[samples['label'] == 0]

    samples_X = samples.drop(['label'], axis=1)
    samples_Y = samples.label

    samples_X = samples_X.loc[:, features]

    # pos_samples_X = pos_samples.drop(['label'], axis=1)
    # pos_samples_Y = pos_samples.label
    #
    # neg_samples_X = neg_samples.drop(['label'], axis=1)
    # neg_samples_Y = neg_samples.label
    #
    # pos_samples_X = pos_samples_X.loc[:, features]
    # neg_samples_X = neg_samples_X.loc[:, features]

    kf_pos = StratifiedKFold(num_folds, random_state=random_state, shuffle=True)

    list_sensitity = []
    list_specifity = []

    list_sensitity_jordan = []
    list_specifity_jordan = []

    list_threshold_prob = []
    list_threshold_prob_via_jordan = []
    list_threshold_value = []
    list_auc = []

    list_acc = []
    list_precision = []
    list_recall = []
    list_f1 = []
    count = 0

    count = 0
    # for pos_train_index, pos_test_index in tqdm(kf_pos.split(pos_samples_X, pos_samples_Y)):
    for train_index, test_index in tqdm(kf_pos.split(samples_X, samples_Y)):
        count += 1
        # if count > 5:
        #     continue

        train_x = samples_X.iloc[train_index, :]
        train_y = samples_Y.iloc[train_index]
        test_x = samples_X.iloc[test_index, :]
        test_y = samples_Y.iloc[test_index]

        # pos_train_x = pos_samples_X.iloc[pos_train_index, :]
        # pos_train_y = pos_samples_Y.iloc[pos_train_index]
        # pos_test_x = pos_samples_X.iloc[pos_test_index, :]
        # pos_test_y = pos_samples_Y.iloc[pos_test_index]
        #
        # neg_train_x = neg_samples_X.iloc[neg_train_index, :]
        # neg_train_y = neg_samples_Y.iloc[neg_train_index]
        # neg_test_x = neg_samples_X.iloc[neg_test_index, :]
        # neg_test_y = neg_samples_Y.iloc[neg_test_index]
        #
        # train_x = pd.concat([pos_train_x, neg_train_x], axis=0)
        # train_y = pd.concat([pos_train_y, neg_train_y], axis=0)
        # test_x = pd.concat([pos_test_x, neg_test_x], axis=0)
        # test_y = pd.concat([pos_test_y, neg_test_y], axis=0)

        train_x.reset_index(drop=True, inplace=True)
        test_x.reset_index(drop=True, inplace=True)
        train_y.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)

        clf.fit(train_x, train_y)

        y_one_hot = label_binarize(test_y, classes=np.arange(2))  # 装换成类似二进制的编码
        print(y_one_hot.shape)
        # print(clf.predict_proba(test_x).shape)
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(
        #     y_one_hot.ravel(), clf.predict_proba(test_x)[:, 1].ravel()
        # )
        # print("false_positive_rate: ", len(false_positive_rate))
        # print("true_positive_rate: ", len(true_positive_rate))
        # print("thresholds: ", len(thresholds))
        if clf.predict_proba(test_x).shape[-1] > 2:
            list_auc.append(roc_auc_score(test_y, clf.predict_proba(test_x), multi_class="ovr"))
        else:
            list_auc.append(
                roc_auc_score(
                    test_y, clf.predict_proba(test_x)[:, 1],
                )
            )
        # list_auc.append(roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1]))
        # list_f1.append(f1_score(test_y, clf.predict(test_x)))

        # sensitity0, specifity0, threshold_prob0 = threshold_prob(
        #     clf, test_x, test_y, sensitivity_threshold=sensitivity_threshold
        # )

        if clf.predict_proba(test_x).shape[-1] > 2:
            sensitity0, specifity0, threshold_prob0 = None, None, None
            sensitity1, specifity1, threshold_prob1 = None, None, None
        else:
            sensitity0, specifity0, threshold_prob0 = threshold_prob(
                clf, test_x, test_y, sensitivity_threshold=sensitivity_threshold
            )
            sensitity1, specifity1, threshold_prob1 = threshold_prob_via_jordan(
                clf, test_x, test_y)


        list_sensitity.append(sensitity0)
        list_specifity.append(specifity0)
        list_threshold_prob.append(threshold_prob0)

        list_sensitity_jordan.append(sensitity1)
        list_specifity_jordan.append(specifity1)
        list_threshold_prob_via_jordan.append(threshold_prob1)

        # 计算：
        if clf.predict_proba(test_x).shape[-1] > 2:
            pred_labels = clf.predict(test_x)
        else:
            pred_labels = (clf.predict_proba(test_x)[:, 1] > threshold_prob1)


        list_acc.append(
            accuracy_score(test_y, pred_labels)
        )
        list_precision.append(
            precision_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)
        )
        list_recall.append(
            recall_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)
        )
        list_f1.append(
            f1_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)
        )

        # print(list_f1)
        # print(list_precision)
        # print(list_recall)

        # 模型存储
        if save_dir:
            clf.save_model(os.path.join(save_dir, f'clf_{count}.model'))


    # list_auc = sorted(list_auc, reverse=True)[2:7]
    # list_sensitity = sorted(list_sensitity, reverse=True)[2:7]
    # list_specifity = sorted(list_specifity, reverse=True)[2:7]
    # list_threshold_prob = list_threshold_prob[2:7]

    return {
        "敏感性": list_sensitity,
        "特异性": list_specifity,
        "敏感性_jordan": list_sensitity_jordan,
        "特异性_jordan": list_specifity_jordan,
        "阈值": list_threshold_prob,
        "阈值_jordan": list_threshold_prob_via_jordan,
        "auc": list_auc,
        "acc": list_acc,
        "precision": list_precision,
        "recall": list_recall,
        "f1": list_f1,

        "平均敏感性": np.mean(list_sensitity) if list_sensitity[0] is not None else None,
        "平均特异性": np.mean(list_specifity)if list_sensitity[0] is not None else None,

        "平均敏感性_jordan": np.mean(list_sensitity_jordan) if list_sensitity[0] is not None else None,
        "平均特异性_jordan": np.mean(list_specifity_jordan) if list_sensitity[0] is not None else None,

        "平均阈值": np.mean(list_threshold_prob) if list_sensitity[0] is not None else None,
        "平均阈值_jordan": np.mean(list_threshold_prob_via_jordan) if list_sensitity[0] is not None else None,
        "平均auc": np.mean(list_auc),
        "auc_标准差": np.std(list_auc),

        "平均acc": np.mean(list_acc),
        "acc_标准差": np.std(list_acc),
        "平均precision": np.mean(list_precision),
        "precision_标准差": np.std(list_precision),
        "平均recall": np.mean(list_recall),
        "recall_标准差": np.std(list_recall),
        "平均f1": np.mean(list_f1),
        "f1_标准差": np.std(list_f1),

        # "平均f1": np.mean(list_f1),
        # "平均阈值原值": np.mean(list_threshold_value) if len(list_threshold_value) > 0 else None,
    }


def train_cv_with_thres_and_sampling(samples,
                        clf,
                        features,
                        num_folds=5,
                        random_state=101,
                        sensitivity_threshold=0.75,
                        save_dir=None,
                        ):
    pos_samples = samples[samples['label'] == 1]
    neg_samples = samples[samples['label'] == 0]

    pos_samples_X = pos_samples.drop(['label'], axis=1)
    pos_samples_Y = pos_samples.label

    neg_samples_X = neg_samples.drop(['label'], axis=1)
    neg_samples_Y = neg_samples.label

    pos_samples_X = pos_samples_X.loc[:, features]
    neg_samples_X = neg_samples_X.loc[:, features]

    kf_pos = StratifiedKFold(10, random_state=random_state, shuffle=True)
    kf_neg = StratifiedKFold(2, random_state=random_state, shuffle=True)

    list_sensitity = []
    list_specifity = []
    list_sensitity_jordan = []
    list_specifity_jordan = []

    list_threshold_prob = []
    list_threshold_prob_via_jordan = []
    list_threshold_value = []
    list_auc = []

    list_acc = []
    list_precision = []
    list_recall = []
    list_f1 = []
    count = 0

    count = 0
    for pos_train_index, pos_test_index in tqdm(kf_pos.split(pos_samples_X, pos_samples_Y)):
        for neg_train_index, neg_test_index in tqdm(kf_neg.split(neg_samples_X, neg_samples_Y)):
            count += 1
            if count > 19:
                continue

            pos_train_x = pos_samples_X.iloc[pos_train_index, :]
            pos_train_y = pos_samples_Y.iloc[pos_train_index]
            pos_test_x = pos_samples_X.iloc[pos_test_index, :]
            pos_test_y = pos_samples_Y.iloc[pos_test_index]

            neg_train_x = neg_samples_X.iloc[neg_train_index, :]
            neg_train_y = neg_samples_Y.iloc[neg_train_index]
            neg_test_x = neg_samples_X.iloc[neg_test_index, :]
            neg_test_y = neg_samples_Y.iloc[neg_test_index]

            train_x = pd.concat([pos_train_x, neg_train_x], axis=0)
            train_y = pd.concat([pos_train_y, neg_train_y], axis=0)
            test_x = pd.concat([pos_test_x, neg_test_x], axis=0)
            test_y = pd.concat([pos_test_y, neg_test_y], axis=0)

            train_x.reset_index(drop=True, inplace=True)
            test_x.reset_index(drop=True, inplace=True)
            train_y.reset_index(drop=True, inplace=True)
            test_y.reset_index(drop=True, inplace=True)

            clf.fit(train_x, train_y)

            y_one_hot = label_binarize(test_y, classes=np.arange(2))  # 装换成类似二进制的编码
            print(y_one_hot.shape)
            # print(clf.predict_proba(test_x).shape)
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(
            #     y_one_hot.ravel(), clf.predict_proba(test_x)[:, 1].ravel()
            # )
            # print("false_positive_rate: ", len(false_positive_rate))
            # print("true_positive_rate: ", len(true_positive_rate))
            # print("thresholds: ", len(thresholds))
            if clf.predict_proba(test_x).shape[-1] > 2:
                list_auc.append(roc_auc_score(test_y, clf.predict_proba(test_x), multi_class="ovr"))
            else:
                list_auc.append(
                    roc_auc_score(
                        test_y, clf.predict_proba(test_x)[:, 1],
                    )
                )
            # list_auc.append(roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1]))
            # list_f1.append(f1_score(test_y, clf.predict(test_x)))

            sensitity0, specifity0, threshold_prob0 = threshold_prob(
                clf, test_x, test_y, sensitivity_threshold=sensitivity_threshold
            )

            list_sensitity.append(sensitity0)
            list_specifity.append(specifity0)
            list_threshold_prob.append(threshold_prob0)

            sensitity1, specifity1, threshold_prob1 = threshold_prob_via_jordan(
                clf, test_x, test_y)
            list_sensitity_jordan.append(sensitity1)
            list_specifity_jordan.append(specifity1)
            list_threshold_prob_via_jordan.append(threshold_prob1)

            # 计算：
            pred_labels = (clf.predict_proba(test_x)[:, 1] > threshold_prob1)
            list_acc.append(
                accuracy_score(test_y, pred_labels)
            )
            list_precision.append(
                precision_score(test_y, pred_labels)
            )
            list_recall.append(
                recall_score(test_y, pred_labels)
            )
            list_f1.append(
                f1_score(test_y, pred_labels)
            )

            # 模型存储
            if save_dir:
                clf.save_model(os.path.join(save_dir, f'clf_{count}.model'))


    # list_auc = sorted(list_auc, reverse=True)[2:7]
    # list_sensitity = sorted(list_sensitity, reverse=True)[2:7]
    # list_specifity = sorted(list_specifity, reverse=True)[2:7]
    # list_threshold_prob = list_threshold_prob[2:7]

    return {
        "敏感性": list_sensitity,
        "特异性": list_specifity,
        "敏感性_jordan": list_sensitity_jordan,
        "特异性_jordan": list_specifity_jordan,
        "阈值": list_threshold_prob,
        "阈值_jordan": list_threshold_prob_via_jordan,
        "auc": list_auc,
        "acc": list_acc,
        "precision": list_precision,
        "recall": list_recall,
        "f1": list_f1,

        "平均敏感性": np.mean(list_sensitity),
        "平均特异性": np.mean(list_specifity),
        "平均敏感性_jordan": np.mean(list_sensitity_jordan),
        "平均特异性_jordan": np.mean(list_specifity_jordan),
        "平均阈值": np.mean(list_threshold_prob),
        "平均阈值_jordan": np.mean(list_threshold_prob_via_jordan),
        "平均auc": np.mean(list_auc),
        "auc_标准差": np.std(list_auc),

        "平均acc": np.mean(list_acc),
        "acc_标准差": np.std(list_acc),
        "平均precision": np.mean(list_precision),
        "precision_标准差": np.std(list_precision),
        "平均recall": np.mean(list_recall),
        "recall_标准差": np.std(list_recall),
        "平均f1": np.mean(list_f1),
        "f1_标准差": np.std(list_f1),

        # "平均f1": np.mean(list_f1),
        # "平均阈值原值": np.mean(list_threshold_value) if len(list_threshold_value) > 0 else None,
    }


def train_cv_for_roc(samples,
                     clf,
                     features,
                     num_folds=5,
                     random_state=101,
                     ):
    # pos_samples = samples[samples['label'] == 1]
    # neg_samples = samples[samples['label'] == 0]

    samples_X = samples.drop(['label'], axis=1)
    samples_Y = samples.label
    samples_X = samples_X.loc[:, features]

    kf_pos = StratifiedKFold(num_folds, random_state=random_state, shuffle=True)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in tqdm(enumerate(kf_pos.split(samples_X, samples_Y))):
        train_x = samples_X.iloc[train_index, :]
        train_y = samples_Y.iloc[train_index]
        test_x = samples_X.iloc[test_index, :]
        test_y = samples_Y.iloc[test_index]

        train_x.reset_index(drop=True, inplace=True)
        test_x.reset_index(drop=True, inplace=True)
        train_y.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)

        clf.fit(train_x, train_y)

        # shape_ = clf.predict_proba(test_x).shape[-1]

        y_one_hot = label_binarize(test_y, classes=np.arange(2))  # 装换成类似二进制的编码
        print(y_one_hot.shape)
        print(clf.predict_proba(test_x).shape)

        fig, ax = plt.subplots()
        viz = plot_roc_curve(clf, test_x, test_y,
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        plt.close()

    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0

    # sorted_ids = sorted(range(len(aucs)), key=lambda x: aucs[x], reverse=True)
    # sorted_ids = sorted_ids[2: 7]
    #
    # tprs = [tprs[i] for i in sorted_ids]
    # aucs = [aucs[i] for i in sorted_ids]

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "aucs": aucs,
        "平均auc": np.mean(aucs),
        "auc_标准差": np.std(aucs),
        "mean_tpr": mean_tpr.tolist(),
        "mean_fpr": mean_fpr.tolist(),
    }

# def train_cv_regressor_with_thres(samples,
#                         clf,
#                         features,
#                         num_folds=5,
#                         random_state=101,
#                         calc_threshold_original_value=False,
#                         interval=None,
#                         margin=0.5e-3,
#                         sensitivity_threshold=0.75
#                         ):
#     # pos_samples = samples[samples['label'] == 1]
#     # neg_samples = samples[samples['label'] == 0]
#
#     samples_X = samples.drop(['label'], axis=1)
#     samples_Y = samples.label
#
#     samples_X = samples_X.loc[:, features]
#
#     kf_pos = StratifiedKFold(num_folds, random_state=random_state, shuffle=True)
#
#     list_sensitity = []
#     list_specifity = []
#     list_threshold_prob = []
#     list_threshold_value = []
#     list_auc = []
#     list_f1 = []
#     list_scores = []
#     count = 0
#     # for pos_train_index, pos_test_index in tqdm(kf_pos.split(pos_samples_X, pos_samples_Y)):
#     for train_index, test_index in tqdm(kf_pos.split(samples_X, samples_Y)):
#         count += 1
#
#         if count > 1:
#             continue
#
#         train_x = samples_X.iloc[train_index, :]
#         train_y = samples_Y.iloc[train_index]
#         test_x = samples_X.iloc[test_index, :]
#         test_y = samples_Y.iloc[test_index]
#
#         train_x.reset_index(drop=True, inplace=True)
#         test_x.reset_index(drop=True, inplace=True)
#         train_y.reset_index(drop=True, inplace=True)
#         test_y.reset_index(drop=True, inplace=True)
#
#         clf.fit(train_x, train_y)
#         scores = clf.score(test_x, test_y)
#         list_scores.append(scores)
#
#         # try:
#         #
#         #     list_auc.append(roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1]))
#         # except Exception as e:
#         #     list_auc.append(roc_auc_score(test_y, clf.predict(test_x)))
#         #
#         # list_f1.append(f1_score(test_y, clf.predict(test_x)))
#
#         # sensitity0, specifity0, threshold_prob0 = threshold_prob(
#         #     clf, test_x, test_y, sensitivity_threshold=sensitivity_threshold)
#
#         # list_sensitity.append(sensitity0)
#         # list_specifity.append(specifity0)
#         # list_threshold_prob.append(threshold_prob0)
#         #
#         # if calc_threshold_original_value:
#         #     thres_final = None
#         #     for thres in tqdm(np.linspace(interval[1], interval[0], 20000)):
#         #         pd_dict = pd.DataFrame([{features[0]: thres}])
#         #         prob = clf.predict_proba(pd_dict)
#         #         prob = prob[:, 1]
#         #
#         #         if - margin < prob - threshold_prob0 < margin:
#         #             thres_final = thres
#         #             break
#         #     print("阈值： ", thres_final)
#         #     list_threshold_value.append(thres_final)
#
#     list_scores = sorted(list_scores, reverse=True)[:5]
#     # list_auc = sorted(list_auc, reverse=True)[:5]
#     # list_sensitity = sorted(list_sensitity, reverse=True)[:5]
#     # list_specifity = sorted(list_specifity, reverse=True)[:5]
#     # list_threshold_prob = list_threshold_prob[:5]
#
#     return {
#         # "敏感性": list_sensitity,
#         # "特异性": list_specifity,
#         # "阈值": list_threshold_prob,
#         "list_scores": list_scores,
#         # "auc": list_auc,
#         # "f1": list_f1,
#         # "平均敏感性": np.mean(list_sensitity),
#         # "平均特异性": np.mean(list_specifity),
#         # "平均阈值": np.mean(list_threshold_prob),
#         # "平均auc": np.mean(list_auc),
#         # "auc_标准差": np.std(list_auc),
#         # "平均f1": np.mean(list_f1),
#         # "平均阈值原值": np.mean(list_threshold_value) if len(list_threshold_value) > 0 else None,
#     }



def train_test_with_thres(train_samples,
                          test_samples,
                        clf,
                        features,
                        random_state=101,
                        sensitivity_threshold=0.75,
                        save_dir=None,
                        ):

    train_samples_X = train_samples.drop(['label'], axis=1)
    train_y = train_samples.label
    train_x = train_samples_X.loc[:, features]

    test_samples_X = test_samples.drop(['label'], axis=1)
    test_y = test_samples.label
    test_x = test_samples_X.loc[:, features]

    train_x.reset_index(drop=True, inplace=True)
    test_x.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)

    clf.fit(train_x, train_y)

    y_one_hot = label_binarize(test_y, classes=np.arange(2))  # 装换成类似二进制的编码
    print(y_one_hot.shape)
    if clf.predict_proba(test_x).shape[-1] > 2:
        auc = roc_auc_score(test_y, clf.predict_proba(test_x), multi_class="ovr")
    else:
        auc = roc_auc_score(
                test_y, clf.predict_proba(test_x)[:, 1],
        )

    if clf.predict_proba(test_x).shape[-1] > 2:
        sensitity0, specifity0, threshold_prob0 = None, None, None
        sensitity1, specifity1, threshold_prob1 = None, None, None
    else:
        sensitity0, specifity0, threshold_prob0 = threshold_prob(
            clf, test_x, test_y, sensitivity_threshold=sensitivity_threshold
        )

        sensitity1, specifity1, threshold_prob1 = threshold_prob_via_jordan(
            clf, test_x, test_y)

    # 计算：
    if clf.predict_proba(test_x).shape[-1] > 2:
        pred_labels = clf.predict(test_x)
    else:
        pred_labels = (clf.predict_proba(test_x)[:, 1] > threshold_prob1)
    acc = accuracy_score(test_y, pred_labels,)
    precision = precision_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)
    recall = recall_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)
    f1 = f1_score(test_y, pred_labels, average="micro" if clf.predict_proba(test_x).shape[-1] > 2 else None)

    return {
        "敏感性": sensitity0,
        "特异性": specifity0,
        "敏感性_jordan": sensitity1,
        "特异性_jordan": specifity1,
        "阈值": threshold_prob0,
        "阈值_jordan": threshold_prob1,
        "auc": auc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }