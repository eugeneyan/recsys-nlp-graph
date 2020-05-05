from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_auc(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend(loc='lower right')

    plt.xlabel('Threshold (0.00 - 1.00)')
    plt.ylabel('Precision / Recall')
    _ = plt.title(title)


def plot_roc(label, score, title):
    fpr, tpr, roc_thresholds = roc_curve(label, score)
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.plot(fpr, tpr, color='b')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    _ = plt.title(title)


def plot_tradeoff(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.step(recall, precision, color='b', label='Precision-Recall Trade-off')
    plt.fill_between(recall, precision, alpha=0.1, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    _ = plt.title(title)


def plot_metrics(df, ylim=None):
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(df.index, df['auc'], label='AUC-ROC', color='black')

    # Plot learning rate resets
    lr_reset_batch = df[df['batches'] == df['batches'].max()]
    for idx in lr_reset_batch.index:
        plt.vlines(idx, df['auc'].min(), 1, label='LR reset (per epoch)',
                   linestyles='--', colors='grey')

    # PLot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    _ = plt.legend(by_label.values(), by_label.keys(), loc='lower right')

    # Tidy axis
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(df['auc'].min() * 1.2, 0.96)
    plt.xlim(0, df.index.max())
    plt.ylabel('AUC-ROC', size=12)
    plt.xlabel('Batches (over 5 epochs)', size=12)
    _ = plt.title('AUC-ROC on sample val set over 5 epochs', size=15)
