import numpy as np
import matplotlib
from imblearn.over_sampling import SMOTE

from sklearn import datasets
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,\
    balanced_accuracy_score, make_scorer, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from inspect import signature

from datetime import datetime
from threading import Thread
from queue import Queue
from multiprocessing import cpu_count
from imblearn import pipeline as imblearn_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.multiclass import unique_labels


aminocode = ["G", "A", "V", "L",
             "I", "M", "P", "F",
             "W", "S", "T", "C",
             "Y", "N", "Q", "D",
             "E", "K", "R", "H"]

polaritypka = [9.78, 9.87, 9.74, 9.74,
               9.76, 9.28, 10.64, 9.31,
               9.41, 9.21, 9.10, 10.70,
               9.21, 8.72, 9.13, 9.90,
               9.47, 9.06, 8.99, 9.33]

polaritybinary = [-1.0, -1.0, -1.0, -1.0,
                  -1.0, -1.0, -1.0, -1.0,
                  -1.0, 1.0, 1.0, -1.0,
                  1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0]

is_positive = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 1.0, 0.0]

is_negative = [0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 1.0,
               1.0, 0.0, 0.0, 0.0]

is_neutral = [1.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0]

kytedoolittle = [-0.4, 1.8, 4.2, 3.8,
                 4.5, 1.9, -1.6, 2.8,
                 -0.9, -0.8, -0.7, 2.5,
                 -1.3, -3.5, -3.5, -3.5,
                 -3.5, -3.9, -4.0, -3.2]

eisenberg = [0.16, 0.25, 0.54, 0.53,
             0.73, 0.26, -0.07, 0.61,
             0.37, -0.26, -0.18, 0.04,
             0.02, -0.64, -0.69, -0.72,
             -0.62, -1.10, -1.80, -0.40]

# scale from http://assets.geneious.com/manual/8.0/GeneiousManualsu41.html
normalized = [0.501, 0.616, 0.825, 0.943,
              0.943, 0.738, 0.711, 1.000,
              0.878, 0.359, 0.450, 0.680,
              0.880, 0.236, 0.251, 0.028,
              0.043, 0.283, 0.000, 0.165]

hydrophobicity = eisenberg          #TODO: change for hydrophobicity (and indicator too!)
hydroindicator = 3                # 1: normalized, 2: kyte-doolittle, 3: eisenberg
avgkytedoolittle = sum(kytedoolittle) / len(kytedoolittle)
avgeisenberg = sum(eisenberg) / len(eisenberg)
avgnormalized = sum(normalized) / len(normalized)
s = max(polaritypka)
normpka = [float(i) / s for i in polaritypka]
polarity = normpka                  #TODO: change for polarity
polarityindicator = 1            # 1: normalized pka, 2: pka, 3: binary
avgnormpka = sum(normpka) / len(normpka)
avgpolaritypka = sum(polaritypka) / len(polaritypka)
avgpolaritybinary = sum(polaritybinary) / len(polaritybinary)

def is_sp(c):
    result = (c == "S" or c == "L" or c == "T")
    return result


def get_acid_index(c):
    result = -1
    # simply stepping through array
    for i in range(0, len(aminocode)):
        if c == aminocode[i]:
            result = i
            break
    return result


def load_fasta(filename):
    print("Started to load %s..." % filename)

    file = open(filename)
    lines = file.readlines()
    seqvecs = []
    seqlabels = []
    aminocounter = 0
    for i in range(0, len(lines), 3):
        # Loop over sequences
        header = lines[i + 0].strip()
        amino = lines[i + 1].strip()
        labels = lines[i + 2].strip()

        # Print mismatch lines
        # if len(amino) != 70:
        #     print("%s: L%d %s" % (filename, i + 2, labels))

        seqvec = []
        for c in amino:
            # Loop over amino acids
            acid_index = get_acid_index(c)
            vec = np.zeros(25) #TODO: add new rows after changing to length 25!
            vec[acid_index] = 1.0
            vec[20] = hydrophobicity[acid_index]
            seqvec.append(vec)
            aminocounter = aminocounter + 1

        #seqvec = np.array(seqvec)
        seqvecs.append(seqvec)

        # Build train result list
        for c in labels:
            if c == "S" or c == "L" or c == "T":
                seqlabels.append(1)
            else:
                seqlabels.append(0)

    #assert len(seqlabels) == len(seqvecs)
    #seqvecs = np.array(seqvecs)
    seqlabels = np.array(seqlabels)

    print("Finished loading %s. (%d aminos loaded)" % (filename, aminocounter))

    return seqvecs, seqlabels, aminocounter


# Number of additional vectors / surrounding AAs
def get_surroundings():
    return 5


def get_columns():
    return 2 * get_surroundings() + 1


# get the number of datasets in one file
def get_amino_count(filename):
    file = open(filename)
    lines = file.readlines()
    aminocounter = 0
    for i in range(0, len(lines), 3):
        amino = lines[i + 1].strip()
        aminocounter += len(amino)
    file.close()
    return aminocounter


def load_fasta_and_create_sliding_window(seqlabels, netinput, offset, filename):
    print("%s: Started to load %s... (to %d)" % (nowstr(), filename, offset))

    surroundings = get_surroundings()
    columns = get_columns()

    file = open(filename)
    lines = file.readlines()
    aminocounter = get_amino_count(filename)

    sp_count = 0

    seqlabelindex = offset
    totalaminoindex = offset
    for i in range(0, len(lines), 3):
        # Loop over sequences
        header = lines[i + 0].strip()
        amino = lines[i + 1].strip()
        labels = lines[i + 2].strip()

        # Print mismatch lines
        # if len(amino) != 70:
        #     print("%s: L%d %s" % (filename, i + 2, labels))

        # load all amino acids
        seqvecs = []
        for c in amino:
            # Loop over amino acids
            acid_index = get_acid_index(c)
            vec = np.zeros(25)
            vec[acid_index] = 1.0
            vec[20] = hydrophobicity[acid_index]
            vec[21] = polarity[acid_index]
            vec[22] = is_positive[acid_index]
            vec[23] = is_negative[acid_index]
            vec[24] = is_neutral[acid_index]
            seqvecs.append(vec)

        # build sliding window matrix for each amino acid and append to list
        for aminoindex in range(0, len(amino)):
            mat = np.zeros((columns, 25))
            for j in range(0, columns):
                seqindex = aminoindex - surroundings + j
                if seqindex in range(0, len(seqvecs)):
                    mat[j] = seqvecs[seqindex]
                else:
                    mat[j] = np.zeros(25)
                    if hydroindicator == 1:
                        mat[j][20] = avgnormalized
                    elif hydroindicator == 2:
                        mat[j][20] = avgkytedoolittle
                    elif hydroindicator == 3:
                        mat[j][20] = avgeisenberg
                    if polarityindicator == 1:
                        mat[j][21] = avgnormpka
                    elif polarityindicator == 2:
                        mat[j][21] = avgpolaritypka
                    elif hydroindicator == 3:
                        mat[j][21] = 0
            flatmatrix = np.zeros(shape=((columns*25) + 1,))
            flatmatrix[:(columns*25):] = mat.flatten()
            flatmatrix[columns*25] = aminoindex/70
            netinput[totalaminoindex] = flatmatrix
            totalaminoindex += 1

        # load result labels
        for c in labels:
            if is_sp(c):
                seqlabels[seqlabelindex] = 1
                sp_count += 1
            else:
                seqlabels[seqlabelindex] = 0
            seqlabelindex += 1

    #assert len(seqlabels) == len(seqvecs)
    #seqvecs = np.array(seqvecs)
    #seqlabels = np.array(seqlabels)

    print("%s: Finished loading %s. (%d amino acids loaded, %d SPs total)" %
          (nowstr(), filename, aminocounter, sp_count))


def random_prediction(filename):
    file = open(filename)
    lines = file.readlines()
    appearances = [0, 0, 0, 0, 0, 0]  # Reihenfolge: S,T,L,I,M,O

    for i in range(0, len(lines), 3):
        # Loop over sequences
        labels = lines[i + 2].strip()

        # Zählt Vorkommen von sachen in labels (IMO oder STL) und speichert abs # in Array
        appearances[0] = appearances[0] + labels.count("S")
        appearances[1] = appearances[1] + labels.count("T")
        appearances[2] = appearances[2] + labels.count("L")
        appearances[3] = appearances[3] + labels.count("I")
        appearances[4] = appearances[4] + labels.count("M")
        appearances[5] = appearances[5] + labels.count("O")

    # Berechne Wahr'keit für IMO bzw. STL
    number_of_labels = appearances[0] + appearances[1] + appearances[2] \
                       + appearances[3] + appearances[4] + appearances[5]
    pIMO = (appearances[3] + appearances[4] + appearances[5])/number_of_labels
    pSTL = (appearances[0] + appearances[1] + appearances[2])/number_of_labels
    probabilites = [0, 0, 0, 0, 0, 0]
    probabilites[0] = appearances[0]/number_of_labels
    probabilites[1] = appearances[1]/number_of_labels
    probabilites[2] = appearances[2]/number_of_labels
    probabilites[3] = appearances[3]/number_of_labels
    probabilites[4] = appearances[4]/number_of_labels
    probabilites[5] = appearances[5]/number_of_labels

    correct_answers = wrong_answers = false_positives = false_negatives = true_positives = true_negatives = 0
    correct_answers_r = wrong_answers_r = false_positives_r = false_negatives_r\
        = true_positives_r = true_negatives_r = 0
    final_number_choice = -1
    alltruelables = np.zeros(shape=(number_of_labels,))
    randomchoices = np.zeros(shape=(number_of_labels,))
    k = 0                                                   # residue counter
    # Noch ein loop über file
    for i in range(0, len(lines), 3):
        labels = lines[i + 2].strip()
        for j in range(0, len(labels)):
            # Möglichkeit 1
            label_choice = str(np.random.choice(a=['S', 'T', 'L', 'I', 'M', 'O'], p=probabilites))
            # print("label choice: " + str(label_choice))
            if label_choice is 'S' or label_choice is 'T' or label_choice is 'L':
                final_number_choice = 1
                randomchoices[k] = 1
            elif label_choice is 'I' or label_choice is 'M' or label_choice is 'O':
                final_number_choice = 0
                randomchoices[k] = 0
            # Möglichkeit 2
            number_choice = 0.8*pSTL + 0.2*(-(1/(1 + (np.exp(-0.5*(j-25))))) + 1)
            # print("number choice: " + str(number_choice))
            # final_number_choice = round(number_choice)

            label = labels[j]
            if (label is 'S' or label is 'T' or label is 'L') and (final_number_choice == 1):
                alltruelables[k] = 1
                true_positives = true_positives + 1
                true_positives_r = true_positives_r + 1
            elif (label is 'I' or label is 'M' or label is 'O') and (final_number_choice == 0):
                alltruelables[k] = 0
                true_negatives = true_negatives + 1
                true_negatives_r = true_negatives_r + 1
            elif (label is 'S' or label is 'T' or label is 'L') and (final_number_choice == 0):
                alltruelables[k] = 1
                false_negatives = false_negatives + 1
                false_negatives_r = false_negatives_r + 1
            elif (label is 'I' or label is 'M' or label is 'O') and (final_number_choice == 1):
                alltruelables[k] = 0
                false_positives = false_positives + 1
                false_positives_r = false_positives_r + 1

            if ((label is 'S' or label is 'T' or label is 'L')and (final_number_choice == 1))\
                    or ((label is 'I' or label is 'M' or label is 'O') and (final_number_choice == 0)):
                correct_answers = correct_answers + 1
                correct_answers_r = correct_answers_r + 1
            else:
                wrong_answers = wrong_answers + 1
                wrong_answers_r = wrong_answers_r + 1
            k += 1

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives_r/(true_positives_r + false_negatives_r)
    # specificity = true_negatives_r/(true_negatives_r + false_positives_r)
    accuracy = (true_positives_r + true_negatives_r)/(true_positives_r + false_positives_r +
                                                      false_negatives_r + true_negatives_r)
    acc = accuracy_score(y_true=alltruelables, y_pred=randomchoices)
    mcc = matthews_corrcoef(y_true=alltruelables, y_pred=randomchoices)
    prec = precision_score(y_true=alltruelables, y_pred=randomchoices)
    rec = recall_score(y_true=alltruelables, y_pred=randomchoices)
    bal_acc = balanced_accuracy_score(y_true=alltruelables, y_pred=randomchoices)
    print("// RANDOM PERFORMANCE: //")
    print("  acc:     " + str(acc))
    print("  mcc:     " + str(mcc))
    print("  prec:    " + str(prec))
    print("  rec:     " + str(rec))
    # print("  spec:  " + str(specificity))
    print("  bal_acc: " + str(bal_acc))


# take input lines and put each dataset into the corresponding file
def extract_splits_to_files(filename, prefix):
    file = open(filename)
    lines = file.readlines()

    # load data
    with open(prefix + "split0file.fasta", 'w+') as split0file, open(prefix + "split1file.fasta", 'w+') as split1file,\
            open(prefix + "split2file.fasta", 'w+') as split2file, open(prefix + "split3file.fasta", 'w+') as split3file,\
            open(prefix + "split4file.fasta", 'w+') as split4file:

        maxnumseqs_persplit = 20000     # TODO: change for tests ( try with 330? )
        split0spnum = split1spnum = split2spnum = split3spnum = split4spnum = 0
        split0no_spnum = split1no_spnum = split2no_spnum = split3no_spnum = split4no_spnum = 0
        for i in range(0, len(lines), 3):
            # Loop over sequences
            header = lines[i + 0].strip()
            amino = lines[i + 1].strip()
            labels = lines[i + 2].strip()

            # Write data to files
            has_sp = False
            if labels.count("S") > 0 or labels.count("L") > 0 or labels.count("T") > 0:
                has_sp = True

            if header.endswith("0") and (split0spnum + split0no_spnum) < maxnumseqs_persplit:
                if (has_sp and split0spnum < maxnumseqs_persplit/2) or\
                        (has_sp is False and split0no_spnum < maxnumseqs_persplit/2):
                    split0file.write(header + "\n")
                    split0file.write(amino + "\n")
                    split0file.write(labels + "\n")
                    if has_sp:
                        split0spnum += 1
                    else:
                        split0no_spnum += 1
            elif header.endswith("1") and (split1spnum + split1no_spnum) < maxnumseqs_persplit:
                if (has_sp and split1spnum < maxnumseqs_persplit / 2) or \
                       (has_sp is False and split1no_spnum < maxnumseqs_persplit / 2):
                    split1file.write(header + "\n")
                    split1file.write(amino + "\n")
                    split1file.write(labels + "\n")
                    if has_sp:
                        split1spnum += 1
                    else:
                        split1no_spnum += 1
            elif header.endswith("2") and (split2spnum + split2no_spnum) < maxnumseqs_persplit:
                if (has_sp and split2spnum < maxnumseqs_persplit / 2) or \
                        (has_sp is False and split2no_spnum < maxnumseqs_persplit / 2):
                    split2file.write(header + "\n")
                    split2file.write(amino + "\n")
                    split2file.write(labels + "\n")
                    if has_sp:
                        split2spnum += 1
                    else:
                        split2no_spnum += 1
            elif header.endswith("3") and (split3spnum + split3no_spnum) < maxnumseqs_persplit:
                if (has_sp and split3spnum < maxnumseqs_persplit / 2) or \
                        (has_sp is False and split3no_spnum < maxnumseqs_persplit / 2):
                    split3file.write(header + "\n")
                    split3file.write(amino + "\n")
                    split3file.write(labels + "\n")
                    if has_sp:
                        split3spnum += 1
                    else:
                        split3no_spnum += 1
            elif header.endswith("4") and (split4spnum + split4no_spnum) < maxnumseqs_persplit:
                if (has_sp and split4spnum < maxnumseqs_persplit / 2) or \
                        (has_sp is False and split4no_spnum < maxnumseqs_persplit / 2):
                    split4file.write(header + "\n")
                    split4file.write(amino + "\n")
                    split4file.write(labels + "\n")
                    if has_sp:
                        split4spnum += 1
                    else:
                        split4no_spnum += 1

            # Check if splits are full
            if (split0spnum + split0no_spnum) >= maxnumseqs_persplit and (split1spnum + split1no_spnum) >=\
                    maxnumseqs_persplit and (split2spnum + split2no_spnum) >= maxnumseqs_persplit and \
                    (split3spnum + split3no_spnum) >= maxnumseqs_persplit and (split4spnum + split4no_spnum) >=\
                    maxnumseqs_persplit:
                break

    print("%s: Extracted %s to 5 splits." % (nowstr(), filename))


# convert the input data from the files to the input format of the MLPClassifier
def create_netinput(allseqvecs):
    print("Creating netinput for %d arrays of vectors" % len(allseqvecs))

    # build sliding window matrices
    mats = []
    surroundings = 5  # Number of additional vectors / surrounding AAs
    # loop over all one-hot encoded vectors

    for seqvecs in allseqvecs:
        for i in range(0, len(seqvecs)):
            columns = 2 * surroundings + 1
            mat = np.zeros((columns, 25))
            # build sliding window
            for j in range(0, columns):
                seqindex = i - surroundings + j
                if seqindex in range(0, len(seqvecs)):
                    columnvec = seqvecs[seqindex]
                else:
                    columnvec = np.zeros(25)
                mat[j] = columnvec
            mats.append(mat)

    print("Converted %d sliding window matrices." % len(mats))

    # flatten matrices
    netinput = np.empty(shape=(len(mats), (11 * 25)))
    for i in range(0, len(mats)):
        netinput[i] = mats[i].flatten()

    print("Converted %d matrices to flat matrices." % len(netinput))

    return netinput


def concatenate_labels(a):
    total_elem_count = 0
    for x in a:
        total_elem_count += len(x)

    result = np.zeros(total_elem_count)
    index = 0
    for arr in a:
        for elem in arr:
            result[index] = elem
            index += 1
    return result


def concatenate_sliding_window_matrices(a):
    total_elem_count = 0
    for x in a:
        total_elem_count += len(x)

    columns = get_columns()
    result = np.zeros((total_elem_count, columns * 25))
    mcounter = 0
    for matarr in a:
        for mat in matarr:
            result[mcounter] = mat
            mcounter += 1
    return result


# queue code from https://docs.python.org/3.5/library/queue.html#module-queue
def worker(queue, threadindex):
    while True:
        item = queue.get(block=True, timeout=None)
        if item is None:
            break
        labels = item[0]
        netinput = item[1]
        offset = item[2]
        filename = item[3]
        load_fasta_and_create_sliding_window(labels, netinput, offset, filename)
        queue.task_done()


def nowstr():
    return str(datetime.now())

def plot_roc_curve(fpr, tpr, thresholds, rocaucscore):
    file = open(file="ros_roc_values.txt", mode="w")
    sizesstring = str(len(fpr)) + " " + str(len(tpr)) + " " + str(len(thresholds))
    file.write(sizesstring + "\n")
    file.write(write_textfiles(fpr) + "\n")
    file.write(write_textfiles(tpr) + "\n")
    file.write(write_textfiles(thresholds) + "\n")
    file.write(str(rocaucscore))


def write_textfiles(array):
    string = ""
    counter = 0
    for element in array:
        string += str(element)
        if counter != len(array) - 1:
            string += " "
        counter += 1
    return string


def plot_precrec_curve(prec, recall, thresholds, average_prec):
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    file = open(file="ros_precrec_values.txt", mode="w")
    sizesstring = str(len(prec)) + " " + str(len(recall)) + " " + str(len(thresholds))
    file.write(sizesstring + "\n")
    file.write(write_textfiles(prec) + "\n")
    file.write(write_textfiles(recall) + "\n")
    file.write(write_textfiles(thresholds) + "\n")
    file.write(str(average_prec))


def plot_confusion_matrix(y_true, y_pred, classes, title):
    file = open(file="ros_confmatrix_values.txt", mode="w")
    sizesstring = str(len(y_true)) + " " + str(len(y_pred))
    file.write(sizesstring + "\n")
    file.write(write_textfiles(y_true) + "\n")
    file.write(write_textfiles(y_pred) + "\n")
    file.write(title)


def main():
    extract_splits_to_files("train_set.fasta", "train")
    extract_splits_to_files("test_set.fasta", "test")

    trainsplitnames = ["trainsplit0file.fasta",
                       "trainsplit1file.fasta",
                       "trainsplit2file.fasta",
                       "trainsplit3file.fasta",
                       "trainsplit4file.fasta"]
    testsplitnames = ["testsplit0file.fasta",
                      "testsplit1file.fasta",
                      "testsplit2file.fasta",
                      "testsplit3file.fasta",
                      "testsplit4file.fasta"]

    # load all train/test amino counts into an array
    trainsplitcounters = []
    trainsplitcountsum = 0
    for splitname in trainsplitnames:
        count = get_amino_count(splitname)
        trainsplitcounters.append(count)
        trainsplitcountsum = trainsplitcountsum + count

    testsplitcounters = []
    testsplitcountsum = 0
    for splitname in testsplitnames:
        count = get_amino_count(splitname)
        testsplitcounters.append(count)
        testsplitcountsum = testsplitcountsum + count

    # create giant matrices (for all ml/net input data)
    columns = get_columns()
    trainnetinput = np.zeros((trainsplitcountsum, columns*25 + 1))
    testnetinput = np.zeros((testsplitcountsum, columns*25 + 1))
    trainsplitlabels = np.zeros((trainsplitcountsum,))
    testsplitlabels = np.zeros((testsplitcountsum,))

    # create borders for the train dataset
    trainborder01 = trainsplitcounters[0]
    trainborder12 = trainborder01 + trainsplitcounters[1]
    trainborder23 = trainborder12 + trainsplitcounters[2]
    trainborder34 = trainborder23 + trainsplitcounters[3]
    trainborder4end = trainborder34 + trainsplitcounters[4]

    # create borders for the test dataset
    testborder01 = testsplitcounters[0]
    testborder12 = testborder01 + testsplitcounters[1]
    testborder23 = testborder12 + testsplitcounters[2]
    testborder34 = testborder23 + testsplitcounters[3]
    testborder4end = testborder34 + testsplitcounters[4]

    trainoffsets = [0, trainborder01, trainborder12, trainborder23, trainborder34]
    testoffsets = [0, testborder01, testborder12, testborder23, testborder34]

    # create worker threads
    queue = Queue()
    threads = []
    workercount = cpu_count()
    print("%s: Loading values on max. %d cores." % (nowstr(), workercount))
    for i in range(0, workercount):
        thread = Thread(target=worker, args=(queue, i))
        thread.start()
        threads.append(thread)

    for i in range(0, len(trainoffsets)):
        item = (trainsplitlabels, trainnetinput, trainoffsets[i], trainsplitnames[i])
        queue.put(item)
        item = (testsplitlabels, testnetinput, testoffsets[i], testsplitnames[i])
        queue.put(item)

    # stop workers
    queue.join()
    for i in range(0, workercount):
        queue.put(None)
    for t in threads:
        t.join()

    # load files into giant matrices
    #load_fasta_and_create_sliding_window(trainsplitlabels, trainnetinput, trainoffsets[0], trainsplitnames[0])
    #load_fasta_and_create_sliding_window(trainsplitlabels, trainnetinput, trainoffsets[1], trainsplitnames[1])
    #load_fasta_and_create_sliding_window(trainsplitlabels, trainnetinput, trainoffsets[2], trainsplitnames[2])
    #load_fasta_and_create_sliding_window(trainsplitlabels, trainnetinput, trainoffsets[3], trainsplitnames[3])
    #load_fasta_and_create_sliding_window(trainsplitlabels, trainnetinput, trainoffsets[4], trainsplitnames[4])

    #load_fasta_and_create_sliding_window(testsplitlabels, testnetinput, testoffsets[0], testsplitnames[0])
    #load_fasta_and_create_sliding_window(testsplitlabels, testnetinput, testoffsets[1], testsplitnames[1])
    #load_fasta_and_create_sliding_window(testsplitlabels, testnetinput, testoffsets[2], testsplitnames[2])
    #load_fasta_and_create_sliding_window(testsplitlabels, testnetinput, testoffsets[3], testsplitnames[3])
    #load_fasta_and_create_sliding_window(testsplitlabels, testnetinput, testoffsets[4], testsplitnames[4])

    print("%s: Finished loading data. (%d train/%d test amino acids loaded into sliding window matrices total)" %
          (nowstr(), trainsplitcountsum, testsplitcountsum))

    # plt.plot(np.arange(0, trainsplitcountsum), trainsplitlabels)
    # plt.show()

    # plt.matshow(np.transpose(trainnetinput))
    # plt.show()

    # create network
    mlp = MLPClassifier(early_stopping=True, validation_fraction=0.2, max_iter=500, learning_rate='adaptive')
    pipeline = imblearn_pipeline.Pipeline([('sampling', RandomOverSampler(sampling_strategy=0.25)), ('classification', mlp)])
    linsvc = LinearSVC(max_iter=1000, random_state=0)
    rfc = RandomForestClassifier(n_estimators=300)

    inner_skf = StratifiedKFold()

    # last hidden layer sizes: (1000,), (10000,) **old values**
    # TODO: classification__ wenn pipeline als estimator
    # TODO: MLP: kein max_depth
    param_grid = {
            "hidden_layer_sizes": [(70,), (100,), (130,)],
            # "learning_rate": ['constant', 'invscaling', 'adaptive']
            # "max_depth": [50, 75, 100]
                  }
    inner_scores = []
    outer_predictions = np.zeros_like(testsplitlabels)
    outer_probascores = np.zeros(shape=(len(testsplitlabels), 2))   # TODO: MLP: shape=(len(testsplitlabels), 2)
    splitted_predictions = [[]]

    for test_i in range(0, 5):
        trainindeces_list = []
        if test_i == 0:
            test_index = np.arange(0, testborder01)
            train_index = np.arange(trainborder01, trainborder4end)
            trainindeces_list.extend((1, 2, 3, 4))
            # former:   = np.arange(trainborder01, len(trainseqvecs))
        elif test_i == 1:
            test_index = np.arange(testborder01, testborder12)
            train_index = np.concatenate([np.arange(0, trainborder01), np.arange(trainborder12, trainborder4end)])
            trainindeces_list.extend((0, 2, 3, 4))
        elif test_i == 2:
            test_index = np.arange(testborder12, testborder23)
            train_index = np.concatenate([np.arange(0, trainborder12), np.arange(trainborder23, trainborder4end)])
            trainindeces_list.extend((0, 1, 3, 4))
        elif test_i == 3:
            test_index = np.arange(testborder23, testborder34)
            train_index = np.concatenate([np.arange(0, trainborder23), np.arange(trainborder34, trainborder4end)])
            trainindeces_list.extend((0, 1, 2, 4))
        elif test_i == 4:
            test_index = np.arange(testborder34, testborder4end)
            train_index = np.arange(0, trainborder34)
            trainindeces_list.extend((0, 1, 2, 3))
        else:
            test_index = []
            train_index = []

        traincount = len(train_index)
        testcount = len(test_index)
        print("%s: Cross validating values... (%d train/%d test elements)" % (nowstr(), traincount, testcount))

        mcc_scorer = make_scorer(matthews_corrcoef)
        gs = GridSearchCV(estimator=mlp, param_grid=param_grid, refit=True, cv=inner_skf, scoring=mcc_scorer,
                          iid=False)    # Frage: Grid Search jedes mal neu????

        #assert len(trainnetinput) == len(trainseqlabels)
        X = trainnetinput[train_index]
        y = trainsplitlabels[train_index]

        #print(*trainsplitlabels, sep='\n')
        #assert len(X) == len(y)

        ros = RandomOverSampler(random_state=0, sampling_strategy=0.25)
        # X_oversampled, y_oversampled = ros.fit_resample(X, y)
        rus = RandomUnderSampler(random_state=0)    # evtl mit replacement=True fuer bootstrap (??)
        # X_undersampled, y_undersampled = rus.fit_resample(X, y)

        gs.fit(X=X, y=y)    #TODO: change to X_oversampled/y_oversampled / X_undersampled/y_undersampled

        print("%s: Cross validation results:" % nowstr())
        print("    Best score:  %f" % gs.best_score_)
        inner_scores.append(gs.best_score_)
        print("    Best params: %s" % str(gs.best_params_))

        outer_predictions[test_index] = gs.predict(testnetinput[test_index])
        outer_probascores[test_index] = gs.predict_proba(testnetinput[test_index])
        #TODO: assumes 2nd column is for positive class
        #TODO: if estimator=linsvc: change gs.predict_proba to decision_functon(X),
        # evtl without transpose and not at index [1] ?
        #splitted_predictions[test_i] = outer_predictions[test_index]

    mean_inner_scores = np.mean(inner_scores)

    tn, fp, fn, tp = confusion_matrix(testsplitlabels, outer_predictions).ravel()
    # np.set_printoptions(precision=2) ?
    class_names = ['not_SP', 'SP']

    # confusion matrix
    plot_confusion_matrix(testsplitlabels, outer_predictions, classes=class_names,
                          title='Confusion matrix (without normalization)')

    outer_probascores = outer_probascores.transpose()

    # ROC-curve
    fpr, tpr, thresholds_roc = roc_curve(testsplitlabels, outer_probascores[1])
    rocaucscore = roc_auc_score(y_true=testsplitlabels, y_score=outer_probascores[1]) #TODO: MLP: outer_probascores[1]
    plot_roc_curve(fpr, tpr, thresholds_roc, rocaucscore)

    # prec-recall curve
    precscore, recscore, thresholds_pr = precision_recall_curve(y_true=testsplitlabels, probas_pred=outer_probascores[1])
    average_precscore = average_precision_score(y_true=testsplitlabels, y_score=outer_probascores[1])
    plot_precrec_curve(precscore, recscore, thresholds_pr, average_precscore)

    print("%s: Results:" % nowstr())
    print("    Mean of inner scores:   " + str(mean_inner_scores))
    print("    Accuracy score (outer): " + str(accuracy_score(testsplitlabels, outer_predictions)))
    print("    MCC score (outer):      " + str(matthews_corrcoef(testsplitlabels, outer_predictions)))
    print("    Prec score (outer):     " + str(precision_score(testsplitlabels, outer_predictions)))
    print("    Recall score (outer):   " + str(recall_score(testsplitlabels, outer_predictions)))
    print("    Confusion Matrix (outer): TN/FP/FN/TP = ")
    print("                            " + str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
    print("    AUC ROC score (outer):  " + str(rocaucscore))
    print("    Balanced ACC  (outer):  " + str(balanced_accuracy_score(testsplitlabels, outer_predictions)))

    bootstrapping = np.zeros((1000, len(outer_predictions)))
    bootstrap_scores = np.zeros((5, 1000))
    chosen_indeces = np.zeros((1000, len(outer_predictions)))
    bootstrap_groundtruth = np.zeros((1000, len(outer_predictions)))
    standard_errors = []        # 0: acc, 1: mcc, 2: prec, 3: rec
    for i in range(0, 1000):
        chosen_indeces[i] = np.random.randint(low=0, high=len(outer_predictions), size=len(outer_predictions))
        for j in range(0, len(outer_predictions)):
            bootstrapping[i][j] = outer_predictions[int(chosen_indeces[i][j])]
            bootstrap_groundtruth[i][j] = testsplitlabels[int(chosen_indeces[i][j])]
        bootstrap_scores[0][i] = accuracy_score(y_true=bootstrap_groundtruth[i], y_pred=bootstrapping[i])
        bootstrap_scores[1][i] = matthews_corrcoef(y_true=bootstrap_groundtruth[i], y_pred=bootstrapping[i])
        bootstrap_scores[2][i] = precision_score(y_true=bootstrap_groundtruth[i], y_pred=bootstrapping[i])
        bootstrap_scores[3][i] = recall_score(y_true=bootstrap_groundtruth[i], y_pred=bootstrapping[i])
        bootstrap_scores[4][i] = balanced_accuracy_score(y_true=bootstrap_groundtruth[i], y_pred=bootstrapping[i])
    standard_errors.append(np.std(a=bootstrap_scores[0]))
    standard_errors.append(np.std(a=bootstrap_scores[1]))
    standard_errors.append(np.std(a=bootstrap_scores[2]))
    standard_errors.append(np.std(a=bootstrap_scores[3]))
    standard_errors.append(np.std(a=bootstrap_scores[4]))

    print("   - SE(acc):     " + str(standard_errors[0]))
    print("   - SE(mcc):     " + str(standard_errors[1]))
    print("   - SE(prec):    " + str(standard_errors[2]))
    print("   - SE(rec):     " + str(standard_errors[3]))
    print("   - SE(bal_acc): " + str(standard_errors[4]))

    # print("    ROC-AUC score (outer):  " +  str(roc_auc_score(testsplitlabels, gs.decision_function)))

    # print("Net statistics:")
    # print("    Iterations: %d" % mlp.n_iter_)
    # print("    Loss:       %f" % mlp.loss_)


if __name__ == "__main__":
    main()
