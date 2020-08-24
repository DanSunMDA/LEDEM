# LEDEM

<ins>**L**</ins>ocating <ins>**E**</ins>nhancers based on <ins>**D**</ins>iverse <ins>**E**</ins>pigenetic <ins>**M**</ins>arks

___

- [LEDEM](#ledem)
  - [About LEDEM](#about-ledem)
  - [Requirements](#requirements)
    - [Installation](#installation)
    - [Command line tools](#command-line-tools)
    - [Python Packages](#python-packages)
  - [Pipeline](#pipeline)
    - [Example dataset](#example-dataset)
    - [Preprocess training dataset](#preprocess-training-dataset)
    - [Preprocess whole-genome dataset for prediction](#preprocess-whole-genome-dataset-for-prediction)
    - [Train an XGBoost model](#train-an-xgboost-model)
    - [Perform whole-genome enhancer prediction](#perform-whole-genome-enhancer-prediction)

## About LEDEM

LEDEM is an XGBoost-based tool that can integrate diverse epigenomic marks (histone modifications, DNA methylation, and chromatin accessibility data) to achieve accurate enhancer prediction results.

In our original paper, to train an XGBoost model and predict enhancers genome-wide, we extract four sets of sequence/epigenomic features for training or whole-genome windows: 1) sequence features, including GC content, CpG density, and the distance to the closest transcription start site; 2) features from ATAC-seq data, including the maximal fold change of chromatin accessibility relative to the local background signals, and the number of peaks; 3) features from whole-genome bisulfite sequencing (WGBS) data, comprising of average fractional methylation levels, and the variation of DNA methylation levels across multiple tissues or cell types; 4) features from histone modifications (maximal fold changes of H3K4me1, H3K4me3, and H3K27ac signals relative to input signals).

## Requirements

Run on Unix/Linux and macOS systems.

### Installation

```bash
git clone https://github.com/DanSunMDA/LEDEM.git
```

LEDEM requires its scripts to be added to the PATH and granted execution permission.

```bash
export PATH=/path/to/LEDEM/scripts/:$PATH
cd /path/to/LEDEM/scripts/
chmod u+x *
```

**For macOS users only**, the 'readarray' command required by LEDEM is not available in Bash 3. Please update Bash to version 4+ using [Homebrew](https://brew.sh/).

```bash
brew install bash
```

If Bash is updated successful, running the following command should not generate any output.

```bash
bash command -v readarray >/dev/null || echo "Cannot find readarray"
```

### Command line tools

|Software|Tested version|Source|
|-|-|:-:|
|BEDTools|v2.29.2|[Download](https://bedtools.readthedocs.io/en/latest/content/installation.html)|
|bigWigToBedGraph|v393|[Download](http://hgdownload.soe.ucsc.edu/admin/exe/)
|Python 3|3.7.5|[Download](https://www.python.org/downloads/)

### Python Packages

|Software|Tested version|Source|
|-|-|:-:|
|NumPy|1.19.1|[Download](https://pypi.org/project/numpy/)|
|Scikit-Learn|0.23.1|[Download](https://pypi.org/project/scikit-learn/)
|XGBoost|1.0.2|[Download](https://pypi.org/project/xgboost/)
|Joblib|0.16.0|[Download](https://pypi.org/project/joblib/)

## Pipeline

The LEDEM pipeline consists of four master commands (\<preprocessTrain>, \<preprocessWG>, \<train>, and \<predict>). See help for more detail:

```bash
ledem.sh -h
```

### Example dataset

The dataset consists of a small number of training examples (2,000 enhancer windows covered by P300, 2,000 promoters windows, and 36,000 [9X] random genomic windows; window size=500-bp) in mouse liver (embryonic day 14.5), as well as epigenetic datasets (histone modification, DNA methylation, ATAC-seq) for both liver embryonic day 14.5 and heart postnatal day 0. The liver dataset is used for training, and heart dataset is used for prediction. To test the program, one can either download the [full dataset](https://www.dropbox.com/s/pxtr8v8zb7xhv1z/LEDEM_test.tar.gz?dl=0) (~7GB) or a subset (chr19 only) of the dataset in our Github "test" folder. All coordinates are based on the mouse mm10 genome assembly.

### Preprocess training dataset

```bash
ledem.sh preprocessTrain
```

\<preprocessTrain\> extracts sequence and epigenetic features from given datasets for training intervals. To extract features, you need to provide the genome FASTA file (with an index file), annotation GTF file, and one or more epigenetic dataset (histone modifications, DNA methylation, and/or chromatin accessibility) with file paths in CSV files.

Training intervals are composed of positive and negative examples in BED format. The fourth column denotes whether specific intervals are positive (1) or negative examples (0).

```
chr1 6256500 6257000 1
chr1 6256800 6257300 1
chr1 6228100 6228600 0
chr1 6240500 6241000 0
```

Below is an example CSV file of histone modifications. Fewer or more histone marks are allowed. BigWig is the required format to store signal values (i.e. fold change over control).

```
H3K4me1,histone_train/Liver_E14.5.H3K4me1.bw
H3K4me3,histone_train/Liver_E14.5.H3K4me3.bw
H3K27ac,histone_train/Liver_E14.5.H3K27ac.bw
```

For DNA methylation, per base fractional methylation level and methylation IQR (measuring variation across multiple tissues/cell types) in BED format can be provided. For convenience, pre-calculated methylation IQR in human and mouse using ENCODE WGBS datasets can be accessed [here](https://www.dropbox.com/s/egx7q337oi5vbxd/LEDEM_methylIQR.tar.gz?dl=0). A CSV file of DNA methylation data looks like below:

```
methyl,methyl_train/Liver_E14.5.methyl.bed
methylIQR,methyl_train/methylIQR.bed
```

Below is a CSV file for ATAC-seq data (can apply to DNase-seq data as well). A fold-change-over-control BigWig file (named 'signal') is required; a BED file containing ATAC-seq peaks (named 'peaks') is not required but can help improve the performance.

```
peaks,atac_train/Liver_E14.5.peaks.bed
signal,atac_train/Liver_E14.5.signal.bw
```

As an example, below is the command for a test run using all features for a small set of trainig intervals.

```bash
cd test
mkdir output

FASTA=/path/to/mm10/FASTA
GTF=/path/to/mm10/GTF
INTERVAL_TRAIN=intervals.bed
HISTONE_TRAIN=histone_train.csv
METHYL_TRAIN=methyl_train.csv
ATAC_TRAIN=atac_train.csv
OUTDIR=output

ledem.sh preprocessTrain -i $INTERVAL_TRAIN -f $FASTA -g $GTF -H $HISTONE_TRAIN -M $METHYL_TRAIN -A $ATAC_TRAIN -p 3 -o $OUTDIR
```

### Preprocess whole-genome dataset for prediction

```bash
ledem.sh preprocessWG
```

 \<preprocessWG> extracts sequence and epigenetic features for whole-genome windows. It first generates windows of user-defined window and slide sizes and then extracts sequence/epigenetic features just like \<preprocessTrain>. Note that \<preprocessWG> and \<preprocessTrain> should use the same set of epigenetic features. It is **recommended** that the window size is set to the same size as training intervals (to shorten the runtime of the test dataset, we uses a larger window size).

A file containig chromosome sizes is required for making genome-wide windows. It is recommended to only list the chromosomes of interest.

```
chr1 195471971
chr10 130694993
chr11 122082543
...
```

An example to extract genome-wide features in mouse heart (postnatal day 0) is as below:

```bash
cd test
FASTA=/path/to/mm10/FASTA
GTF=/path/to/mm10/GTF
CHROMINFO=/path/to/mm10/chromSize
HISTONE_PREDICT=histone_predict.csv
METHYL_PREDICT=methyl_predict.csv
ATAC_PREDICT=atac_predict.csv
OUTDIR=output

ledem.sh preprocessWG -w 1000 -s 500 -f $FASTA -g $GTF -c $CHROMINFO -H $HISTONE_PREDICT -M $METHYL_PREDICT -A $ATAC_PREDICT -p 3 -o $OUTDIR
```

### Train an XGBoost model

```bash
ledem.sh train
```

 \<train> trains an XGBoost using the Scikit-Learn wrapper interface for XGBoost ([more info](https://xgboost.readthedocs.io/en/latest/python/python_api.html)). The input is a matrix in CSV format that is obtained by \<preprocessTrain>.

To train a model under default hyperparameter settings:

```bash
cd test
INPUT=$OUTDIR/train.csv
OUTDIR=output

ledem.sh train -i $INPUT -o $OUTDIR
```

To train a model using using user-defined hyperparameters:

```bash
ledem.sh train -i $INPUT -x n_estimators=300,learning_rate=0.05,max_depth=10,n_jobs=8 -o $OUTDIR
```

To train a model with hyperparameter tuning by RandomizedSearchCV (takes significantly more time):

```bash
ledem.sh train -i $INPUT -x n_jobs=2 -t --n_iter 20 --n_jobs 4 -o $OUTDIR
```

### Perform whole-genome enhancer prediction

```bash
ledem.sh predict
```

 \<predict> predicts enhancers at the genome-wide scale. It takes in a matrix in CSV format and windows obtained by \<preprocessWG>, as well as a trained XGBoost model and the scaler used for training data standardization obtained by \<train>. Overlapping enhancer windows and windows within -d \<DISTANCE> are merged to obtain the final set of enhancers.

```bash
cd test
INPUT=$OUTDIR/wg.csv
BED=$OUTDIR/windows.bed
MODEL=$OUTDIR/model.joblib.dat
SCALER=$OUTDIR/scaler.joblib.dat
DISTANCE=500
OUTDIR=output

ledem.sh predict -i $INPUT -b $BED -m $MODEL -s $SCALER -d $DISTANCE -o $OUTDIR
```

Two output files are generated ('scores.bed' and 'enhancers.bed'). 'scores.bed' contains coordinates of windows (1st-3rd columns), scores for negative windows (4th column), and scores for positive windows (5th column).  

```
chr1    5081000 5082000 9.998142719268798828e-01 1.857320748968049884e-04
chr1    5081500 5082500 9.998005628585815430e-01 1.994531776290386915e-04
chr1    5082000 5083000 3.092803359031677246e-01 6.907196640968322754e-01
```

'enhancer.bed' includes the coordinates of predicted enhancers (a score cutoff of 0.5 is used to extract positive windows).
