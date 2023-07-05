import numpy as np
# TODO: new plot object (prec-rec curve ohne roc-curve), evtl so:?
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
from inspect import signature
from sklearn.metrics import confusion_matrix


def plot_precrec_curve(filename):
    # Regenerate Data
    file = open(filename)
    lines = file.readlines()
    seperator = ' '
    sizes = np.zeros(shape=(3,), dtype=int)
    sizes = get_values(sizes, lines[0], seperator)
    prec = np.zeros(shape=(sizes[0],))
    recall = np.zeros(shape=(sizes[1],))
    thresholds = np.zeros(shape=(sizes[2],))
    prec = get_values(prec, lines[1], seperator)
    recall = get_values(recall, lines[2], seperator)
    thresholds = get_values(thresholds, lines[3], seperator)
    average_prec = get_values(None, lines[4], seperator)[0]

    # Plot Curve
    plt = plt1
    plt.rcParams.update({'font.size': 13})
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, prec, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, prec, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: Average Precision ={0:0.2f}'.format(
        average_prec))
    plt.savefig(fname="final_precrec_curve.png", transparent=False)


def get_values(array, line, seperator):
    valstrings = line.split(seperator)
    if array is not None:
        index = 0
        for valstr in valstrings:
            val = float(valstr)
            array[index] = val
            index += 1
    else:
        array = np.zeros(shape=(1,))
        array[0] = float(valstrings[0])
    return array


def plot_roc_curve(filename):
    # Regenerate Data
    file = open(filename)
    lines = file.readlines()
    seperator = ' '
    sizes = np.zeros(shape=(3,), dtype=int)
    sizes = get_values(sizes, lines[0], seperator)
    fpr = np.zeros(shape=(sizes[0],))
    tpr = np.zeros(shape=(sizes[1],))
    thresholds = np.zeros(shape=(sizes[2],))
    fpr = get_values(fpr, lines[1], seperator)
    tpr = get_values(tpr, lines[2], seperator)
    thresholds = get_values(thresholds, lines[3], seperator)
    rocaucscore = get_values(None, lines[4], seperator)[0]

    # Plot Curve
    plt = plt2
    plt.rcParams.update({'font.size': 13})
    plt.figure()
    lw = 2 # linewidth maybe
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f' % rocaucscore)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylabel([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(fname="final_roc_curve.png", transparent=False) # alt: plt.show


def draw_conf_matrix(filename):
    # Regenerate Data
    file = open(filename)
    lines = file.readlines()
    seperator = ' '
    sizes = np.zeros(shape=(2,), dtype=int)
    sizes = get_values(sizes, lines[0], seperator)
    y_true = np.zeros(shape=(sizes[0],))
    y_pred = np.zeros(shape=(sizes[1],))
    y_true = get_values(y_true, lines[1], seperator)
    y_pred = get_values(y_pred, lines[2], seperator)
    title = lines[3]                                    # needed?

    # Draw Confusion Matrix
    plt = plt3
    plt.rcParams.update({'font.size': 18})
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    normalize = True    #TODO: set whether normalized
    cmap = plt.cm.Blues
    # if not title:
    if normalize:
        title = 'Normalized confusion matrix (CM)'
    else:
        title = 'CM without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    tempClasses = np.ndarray(shape=(2,), dtype='object')
    tempClasses[0] = 'NO_SP'
    tempClasses[1] = 'SP'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=tempClasses, yticklabels=tempClasses,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.axis("scaled")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(fname="final_n_confmatrix.png", transparent=False)  # plt.show()


def main():
    # plot_precrec_curve("plot_data/final_precrec_values.txt")
    plot_roc_curve("plot_data/final_roc_values.txt")
    # draw_conf_matrix("plot_data/final_confmatrix_values.txt")


if __name__ == '__main__':
   main()

