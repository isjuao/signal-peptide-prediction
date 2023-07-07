# Signal peptide prediction

This aim of this project was building a simple yet effective machine learning (ML) model to predict signal peptides 
(SPs) from the amino acid protein sequence.

SPs are an important subject to current research in the production of
recombinant proteins. Computationally predicting SPs can improve the
efficiency of the process.

The final paper is available as a .pdf file in this repository.

## Background

Signal peptides (SPs) are cleavable amino acid sequences that are
commonly located at the N-terminal region of newly nascent secretory
proteins. Their main task is to direct their individual protein into or across
a cellular membrane before being removed during translocation through
the membrane.

### Motivation of predicting signal peptides

In recent years research has been facing a growing demand for biotherapeutics, in
particular for recombinant proteins. Production of the latter however remains a challenge, as proteolytic
processing and incorrect protein folding are among many issues which lead to a loss in final product
quantity. A possible way of improving this common situation is provided through increasing the protein
translocation rate and secretion efficiency by tuning the respective signal peptide. Not only are these
N-terminal sequences involved in the production of medication, they also serve as targets of drugs. A
promising example are specifically designed therapies which tackle the signal peptides of proteins
originating from malaria parasites. The difference between their signal peptides and the human
equivalent is the necessary condition enabling this emerging method of battling the malaria disease.
Hence, computational prediction of signal peptides in protein sequences has become of great interest
and numerous algorithms have been developed.

## Implementation

The [scikit-learn](https://scikit-learn.org/stable/) library was used for ML model development.
[Imbalanced-learn](https://imbalanced-learn.org/stable/index.html) supplied 
necessary tools for imbalance correction. 

### Data

The datasets used for this project are the training set and the test set
(available as .fasta files) that were developed for training and
benchmarking of [SignalP 5.0](https://doi.org/10.1038/s41587-019-0036-z).

In order to keep the redundancy found in the datasets provided by SignalP 5.0 as low as possible, we implemented cross-validation
**manually** to preserve the partitions already given by the datasets. Its **nested** form ensures a maximized
use of both datasets. The beneficial effect of the manual implementation of cross-validation becomes
evident in the results.


