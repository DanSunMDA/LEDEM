#!/usr/bin/env bash

### Set variables and default behaviors ###
version="1.0.0"
random=$RANDOM$RANDOM
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

### Functions ###
## Functions to print help
function generalHelp() {
	echo "LEDEM "$version
	echo "Usage: ledem.sh <CMD> [arguments]"
	echo ""
	echo "<CMD> can be one of:"
	echo ""
	echo "preprocessTrain                  Extracts features for training intervals"
	echo "preprocessWG                     Extracts features for whole-genome windows"
	echo "train                            Trains an XGBoost model"
	echo "predict                          Performs a whole-genome prediction of enhancers based on the trained model"
	echo ""
	exit 1
}

function preprocessTrainHelp() {
	echo "LEDEM "$version
	echo "Extracts features for training intervals"
	echo "Usage: ledem.sh preprocessTrain [arguments]"
	echo ""
	echo "Required arguments:"
	echo "-i, --interval <BED>             An interval file with labels (format: chromosome  start  end  label)"
	echo "-f, --fasta <FASTA>              A genome FASTA file"
	echo "-g, --GTF <GTF>                  A GTF annotation file"

	echo ""
	echo "Optional sequence feature arguments:"
	echo "-C, --conserv <BED>              A BED file with conservation score as the 4th column"
	echo "-F, --motif <BED>                A BED file containing a motif per line"

	echo ""
	echo "Epigenetic feature arguments (at least one should be specified):"
	echo "-H, --histone <CSV>              A CSV file containing paths to histone modification files"
	echo "-M, --methyl <CSV>               A CSV file containing paths to DNA methylation files"
	echo "-A, --atac <CSV>                 A CSV file containing paths to ATAC-seq files"

	echo ""
	echo "Optional arguments:"
	echo "-o, --outdir <DIR>               The path to the output directory (default: the working directory)"
	echo "-p, --process <int>              The number of processes (default: 1)"
	echo "-h, --help                       Prints this help message"
	exit 1
}

function preprocessWGHelp() {
	echo "LEDEM "$version
	echo "Extracts features for whole-genome windows"
	echo "Usage: ledem.sh preprocessWG [arguments]"
	echo ""
	echo "Required arguments:"
	echo "-f, --fasta <FASTA>              A genome FASTA file"
	echo "-g, --GTF <GTF>                  A GTF annotation file"
	echo "-c <CHROMINFO>                   A chromosome size info file (format: chromosome  size)"

	echo ""
	echo "Optional sequence feature arguments:"
	echo "-C, --conserv <BEDGRAPH>         A conservation score file (format: chromosome start end score)"
	echo "-F, --motif <BED>                A BED file with a motif per line"

	echo ""
	echo "Epigenetic feature arguments (should be the same set of features as in preprocessTrain):"
	echo "-H, --histone <CSV>              A CSV file containing paths to histone modification files"
	echo "-M, --methyl <CSV>               A CSV file containing paths to DNA methylation files"
	echo "-A, --atac <CSV>                 A CSV file containing paths to ATAC-seq files"

	echo ""
	echo "Optional arguments:"
	echo "-w <int>                         Window size (default: 500bp)"
	echo "-s <int>                         Slide size (default: 100bp)"
	echo "-o, --outdir <DIR>               The path to the output directory (default: the working directory)"
	echo "-p, --process <int>              The number of processes (default: 1)"
	echo "-h, --help                       Prints this help message"
	exit 1
}

function trainHelp() {
	echo "LEDEM "$version
	echo "Trains an XGBoost model in Python"
	echo "Usage: ledem.sh train -i INPUT [arguments]"
	echo ""
	echo "Required arguments:"
	echo "-i, --input <CSV>                The input CSV file prepared by preprocessTrain"
	echo ""
	echo "XGBoost arguments:"
	echo "-x, --xgboost <PARAMS>           Parameters to pass to XGBClassifier for training (default: default parameters set by XGBoost)"
	echo "                                 When multiple parameters are specified, they need to be separated by commas (e.g. n_jobs=8,n_estimators=400,learning_rate=0.01)"
	echo "                                 PARAMS, except 'n_jobs=N', will be omitted if -t is set to true"
	echo "-t, --tuning                     Performs hyperparameter tuning by RandomizedSearchCV (default: not on)"
	echo "--n_iter                         For tuning, the number of parameter settings to sample (default: 100). Requires -t to be turned on"
	echo "--n_jobs                         For tuning, the number of jobs to run in parallel (default: 1). Requires -t to be turned on"
	echo "                                 Note that the total number of processors used is n_jobs(XGBClassifier) * n_jobs(RandomizedSearchCV)"
	echo ""
	echo "Optional arguments:"
	echo "-o, --outdir <DIR>               The path to the output directory (default: the working directory)"
	echo "-h, --help                       Prints this help message"
	exit 1
}

function predictHelp() {
	echo "LEDEM "$version
	echo "Performs a whole-genome prediction of enhancers based on the trained model in Python"
	echo "Usage: ledem.sh predict -i INPUT [arguments]"
	echo ""
	echo "Required arguments:"
	echo "-i, --input <CSV>                The input CSV file prepared by preprocessWG"
	echo "-b, --bed <BED>                  Windows prepared by preprocessWG in BED format"
	echo "-m, --model <DAT>                The XGBoost model (generated by train)"
	echo "-s, --scaler <DAT>               The scaler used for training dataset (generated by train)"
	echo ""
	echo "Optional arguments:"
	echo "-d, --distance <int>             Merges nearby enhancers if within X bp (default: 100)"
	echo "-o, --outdir <DIR>               The path to the output directory (default: the working directory)"
	echo "-h, --help                       Prints this help message"
	exit 1
}

## Functions to get options
function preprocessTrain() {
	if [ $# -eq 0 ]; then
		preprocessTrainHelp
	fi

	while test $# -gt 0; do
		case "$1" in
		-h | --help)
			preprocessTrainHelp
			;;
		-i | --interval)
			shift
			export INTERVAL=$(realpath $1)
			shift
			;;
		-f | --fasta)
			shift
			export FASTA=$(realpath $1)
			shift
			;;
		-g | --gtf)
			shift
			export GTF=$(realpath $1)
			shift
			;;
		-H | --histone)
			shift
			export HISTONE=$(realpath $1)
			shift
			;;
		-M | --methyl)
			shift
			export METHYL=$(realpath $1)
			shift
			;;
		-A | --atac)
			shift
			export ATAC=$(realpath $1)
			shift
			;;
		-C | --conserv)
			shift
			export CONSERV=$(realpath $1)
			shift
			;;
		-F | --motif)
			shift
			export MOTIF=$(realpath $1)
			shift
			;;
		-o | --outdir)
			shift
			export OUTDIR=$(realpath $1)
			shift
			;;
		-p | --process)
			shift
			export PROCESS=$1
			shift
			;;

		*)
			echo "ERROR: Unrecognized argument "$1
			echo ""
			preprocessTrainHelp
			;;
		esac
	done
}

function preprocessWG() {
	if [ $# -eq 0 ]; then
		preprocessWGHelp
	fi

	while test $# -gt 0; do
		case "$1" in
		-h | --help)
			preprocessWGHelp
			;;
		-f | --fasta)
			shift
			export FASTA=$(realpath $1)
			shift
			;;
		-g | --gtf)
			shift
			export GTF=$(realpath $1)
			shift
			;;
		-c)
			shift
			export CHROMINFO=$(realpath $1)
			shift
			;;
		-H | --histone)
			shift
			export HISTONE=$(realpath $1)
			shift
			;;
		-M | --methyl)
			shift
			export METHYL=$(realpath $1)
			shift
			;;
		-A | --atac)
			shift
			export ATAC=$(realpath $1)
			shift
			;;
		-C | --conserv)
			shift
			export CONSERV=$(realpath $1)
			shift
			;;
		-F | --motif)
			shift
			export MOTIF=$(realpath $1)
			shift
			;;
		-w)
			shift
			export WINDOW_SIZE=$1
			shift
			;;
		-s)
			shift
			export SLIDE_SIZE=$1
			shift
			;;
		-o | --outdir)
			shift
			export OUTDIR=$(realpath $1)
			shift
			;;
		-p | --process)
			shift
			export PROCESS=$1
			shift
			;;
		*)
			echo "ERROR: Unrecognized argument "$1
			echo ""
			preprocessWGHelp
			;;
		esac
	done
}

function train() {
	if [ $# -eq 0 ]; then
		trainHelp
	fi

	while test $# -gt 0; do
		case "$1" in
		-h | --help)
			trainHelp
			;;
		-i | --input)
			shift
			export INPUT=$(realpath $1)
			shift
			;;
		-x | --xgboost)
			shift
			export PARAMS=$1
			shift
			;;
		-t | --tuning)
			export TUNING='true'
			shift
			;;
		--n_iter)
			shift
			export n_iter=$1
			shift
			;;
		--n_jobs)
			shift
			export n_jobs=$1
			shift
			;;
		-o | --outdir)
			shift
			export OUTDIR=$(realpath $1)
			shift
			;;
		*)
			echo "ERROR: Unrecognized argument "$1
			echo ""
			trainHelp
			;;
		esac
	done
}

function predict() {
	if [ $# -eq 0 ]; then
		predictHelp
	fi

	while test $# -gt 0; do
		case "$1" in
		-h | --help)
			predictHelp
			;;
		-i | --input)
			shift
			export INPUT=$(realpath $1)
			shift
			;;
		-b | --bed)
			shift
			export BED=$(realpath $1)
			shift
			;;
		-m | --model)
			shift
			export MODEL=$(realpath $1)
			shift
			;;
		-s | --scaler)
			shift
			export SCALER=$(realpath $1)
			shift
			;;
		-d | --distance)
			shift
			export DISTANCE=$1
			shift
			;;
		-o | --outdir)
			shift
			export OUTDIR=$(realpath $1)
			shift
			;;
		*)
			echo "ERROR: Unrecognized argument "$1
			echo ""
			predictHelp
			;;
		esac
	done
}

## Misc. helper functions
function realpath() {
	[[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

function checkDIR() {
	if [[ ! -d "$1" ]]; then
		echo "ERROR: "$1" does not exist"
		exit 1
	fi
	if [[ ! -w "$1" ]]; then
		echo "ERROR: "$1" is not writable"
		exit 1
	fi
}

function checkFILE() {
	if [[ ! -f "$1" ]]; then
		echo "ERROR: Cannot open "$1
		exit 1
	fi
	if [[ ! -s "$1" ]]; then
		echo "ERROR: "$1" is empty"
		exit 1
	fi
}

function checkCSV() {
	awk -F "," '{if (NF < 2) {print "ERROR:",FILENAME,"is not a valid CSV file " ; exit 1}}' $1
	if [[ "$?" == 1 ]]; then
		exit 1
	fi
}

function checkBED() {
	awk -F '\t' '{if (NF < 3) {exit 1}}' $1
	if [[ "$?" != 0 ]]; then
		echo "ERROR: Invalid BED file (at least 3 columns expected) for "$1
		exit 1
	fi
}

function checkGTF {
	awk -F '\t' '{if (NF != 9 && $0 !~ "^#") {exit 1}}' $1
	if [[ "$?" != 0 ]]; then
		echo "ERROR: Invalid GTF file "$1" (9 columns expected)"
		exit 1
	fi
}

function checkINTEGER() {
	if [[ ! "$2" =~ ^[0-9]+ || ! "$2" -gt 0 ]]; then
		echo "ERROR: $1 must be followed by a positive integer"
		exit 1
	fi
}

function checkTools() {
	for tool in trim_galore samtools hisat2 stringtie; do
		command -v ${tool} >/dev/null 2>&1 || {
			echo >&2 ${tool}" is missing (required in the PATH)"
			toolMissing=1
		}
	done
}

## Functions to extract features
function extractFeatures() {
	## Extract sequence features
	echo "Extracting sequence features..."
	awk '$3 == "gene"' $GTF | grep 'protein_coding\|lncRNA' | awk 'BEGIN{OFS="\t"}{if ($7 == "+") print $1,$4-1,$4,$10,$6,$7; else print $1,$5-1,$5,$10,$6,$7}' |
		sed 's/"//g' | sed 's/;//' | sort -k 1,1 -k 2,2n >$TMP/TSS.bed
	TSS=$TMP/TSS.bed

	((k = k % PROCESS))
	((k++ == 0)) && wait
	echo "Calculating GC content"
	bedtools nuc -fi $FASTA -bed $INTERVAL -C -pattern 'cg' | sed 1d | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$5*100}' | cut -f 4 >$TMP/GC_content.dat &
	((k = k % PROCESS))
	((k++ == 0)) && wait
	echo "Calculating CpG density"
	bedtools nuc -fi $FASTA -bed $INTERVAL -C -pattern 'cg' | sed 1d | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$13/$12 * 200}' | cut -f 4 >$TMP/CpG_density.dat &
	((k = k % PROCESS))
	((k++ == 0)) && wait
	echo "Calculating the distance to the nearest TSS"
	bedtools closest -a $INTERVAL -b $TSS -d -t first | cut -f 10 >$TMP/TSS_dist.dat &
	wait

	paste -d ',' $TMP/summary.csv $TMP/GC_content.dat $TMP/CpG_density.dat $TMP/TSS_dist.dat >$TMP/summary2.csv
	mv $TMP/summary2.csv $TMP/summary.csv
	echo ""

	if [[ ! -z "$CONSERV" ]]; then
		echo "Calculating conservation scores..."
		bedtools map -a $INTERVAL -b $CONSERV -c 4 -o mean | sed 's/\.$//' | cut -f 4 >$TMP/conserv.dat

		paste -d ',' $TMP/summary.csv $TMP/conserv.dat >$TMP/summary2.csv
		mv $TMP/summary2.csv $TMP/summary.csv
		echo ""
	fi

	if [[ ! -z "$MOTIF" ]]; then
		echo "Counting number of motifs..."
		bedtools map -a $INTERVAL -b $MOTIF -c 1 -o count | cut -f 4 >$TMP/motif.dat

		paste -d ',' $TMP/summary.csv $TMP/motif.dat >$TMP/summary2.csv
		mv $TMP/summary2.csv $TMP/summary.csv
		echo ""
	fi

	## Extract histone modification features
	if [[ ! -z "$HISTONE" ]]; then
		echo "Extracting features from histone modification data..."
		readarray -t HISTONE_NAMES < <(cut -d ',' -f 1 $HISTONE)
		readarray -t HISTONE_PATHS < <(cut -d ',' -f 2 $HISTONE)

		for ((i = 0; i < ${#HISTONE_NAMES[@]}; i++)); do
			HISTONE_PATH=${HISTONE_PATHS[$i]}
			checkFILE $HISTONE_PATH
		done
		for ((i = 0; i < ${#HISTONE_NAMES[@]}; i++)); do
			((j = j % PROCESS))
			((j++ == 0)) && wait
			HISTONE_NAME=${HISTONE_NAMES[$i]}
			HISTONE_PATH=${HISTONE_PATHS[$i]}
			echo "Processing "$HISTONE_NAME

			bigWigToBedGraph $HISTONE_PATH $TMP/$HISTONE_NAME.bdg &&
				bedtools map -a $INTERVAL -b $TMP/$HISTONE_NAME.bdg -c 4 -o max | sed 's/\.$//' | cut -f 4 >$TMP/$HISTONE_NAME.dat &&
				rm $TMP/$HISTONE_NAME.bdg &
		done
		wait

		for ((i = 0; i < ${#HISTONE_NAMES[@]}; i++)); do
			HISTONE_NAME=${HISTONE_NAMES[$i]}
			paste -d ',' $TMP/summary.csv $TMP/$HISTONE_NAME.dat >$TMP/summary2.csv
			mv $TMP/summary2.csv $TMP/summary.csv
		done
		echo ""
	fi

	# Extract DNA methylation features
	if [[ ! -z "$METHYL" ]]; then
		echo "Extracting features from WGBS data..."
		readarray -t METHYL_NAMES < <(cut -d ',' -f 1 $METHYL)
		readarray -t METHYL_PATHS < <(cut -d ',' -f 2 $METHYL)

		for ((i = 0; i < ${#METHYL_NAMES[@]}; i++)); do
			METHYL_PATH=${METHYL_PATHS[$i]}
			checkFILE $METHYL_PATH
		done
		for ((i = 0; i < ${#METHYL_NAMES[@]}; i++)); do
			((j = j % PROCESS))
			((j++ == 0)) && wait
			METHYL_PATH=${METHYL_PATHS[$i]}
			METHYL_NAME=${METHYL_NAMES[$i]}
			echo "Processing "$METHYL_NAME
			bedtools map -a $INTERVAL -b $METHYL_PATH -c 4 -o mean | sed 's/\.$//' | cut -f 4 >$TMP/$METHYL_NAME.dat &
		done
		wait

		for ((i = 0; i < ${#METHYL_NAMES[@]}; i++)); do
			METHYL_NAME=${METHYL_NAMES[$i]}
			paste -d ',' $TMP/summary.csv $TMP/$METHYL_NAME.dat >$TMP/summary2.csv
			mv $TMP/summary2.csv $TMP/summary.csv
		done
		echo ""
	fi

	## Extract ATAC-seq features
	if [[ ! -z "$ATAC" ]]; then
		echo "Extracting features from ATAC-seq data..."
		readarray -t ATAC_NAMES < <(cut -d ',' -f 1 $ATAC)
		readarray -t ATAC_PATHS < <(cut -d ',' -f 2 $ATAC)

		good=0
		for ((i = 0; i < ${#ATAC_NAMES[@]}; i++)); do
			ATAC_PATH=${ATAC_PATHS[$i]}
			checkFILE $ATAC_PATH
		done
		for ((i = 0; i < ${#ATAC_NAMES[@]}; i++)); do
			ATAC_NAME=${ATAC_NAMES[$i]}
			ATAC_PATH=${ATAC_PATHS[$i]}
			if [[ "$ATAC_NAME" == "signal" ]]; then
				echo "Processing "$ATAC_NAME
				good=1
				bigWigToBedGraph $ATAC_PATH $TMP/atac_signal.bdg
				bedtools map -a $INTERVAL -b $TMP/atac_signal.bdg -c 4 -o max | sed 's/\.$//' | cut -f 4 >$TMP/atac_signal.dat
				rm $TMP/atac_signal.bdg
				paste -d ',' $TMP/summary.csv $TMP/atac_signal.dat >$TMP/summary2.csv
				mv $TMP/summary2.csv $TMP/summary.csv
			fi
			if [[ "$ATAC_NAME" == "peaks" ]]; then
				echo "Processing "$ATAC_NAME
				checkFILE $ATAC_PATH
				bedtools map -a $INTERVAL -b $ATAC_PATH -c 1 -o count | cut -f 4 >$TMP/atac_numPeaks.dat
				paste -d ',' $TMP/summary.csv $TMP/atac_numPeaks.dat >$TMP/summary2.csv
				mv $TMP/summary2.csv $TMP/summary.csv
			fi
		done
		if [[ "$good" -eq 0 ]]; then
			echo "ERROR: The ATAC-seq signal file (named as 'signal' in the CSV file) is required"
			exit 1
		fi
		echo ""
	fi
}

### Pipeline Main ###
if [ $# -eq 0 ]; then
	generalHelp
else
	case "$1" in
	preprocessTrain)
		shift
		preprocessTrain $@
		CMD="preprocessTrain"
		;;
	preprocessWG)
		shift
		preprocessWG $@
		CMD="preprocessWG"
		;;
	train)
		shift
		train $@
		CMD="train"
		;;
	predict)
		shift
		predict $@
		CMD="predict"
		shift
		;;
	*)
		echo "ERROR: A valid CMD argument is required"
		echo ""
		generalHelp
		;;
	esac
fi

if [[ "$CMD" == "preprocessTrain" ]]; then
	## Set defaults and checks
	if [[ -z "$OUTDIR" ]]; then
		OUTDIR=$(pwd)
	fi

	if [[ -z "$PROCESS" ]]; then
		PROCESS=1
	fi

	checkDIR $OUTDIR

	if [[ -z "$INTERVAL" ]]; then
		echo "ERROR: Please provide an interval file in BED format by -i"
		echo ""
		preprocessTrainHelp
	else
		checkBED $INTERVAL
	fi
	
	if [[ -z "$FASTA" ]]; then
		echo "ERROR: Please provide a genome sequence file in FASTA format by -f"
		echo ""
		preprocessTrainHelp
	fi
	
	if [[ -z "$GTF" ]]; then
		echo "ERROR: Please provide an annotation GTF file by -g"
		echo ""
		preprocessTrainHelp
	else
		checkGTF $GTF
	fi
	checkFILE $INTERVAL
	checkFILE $FASTA
	checkFILE $GTF

	if [[ -z "$HISTONE" && -z "$METHYL" && -z "$ATAC" ]]; then
		echo "ERROR: At least one of -H, -M, and -A should be specified"
		echo ""
		preprocessTrainHelp
	fi
	if [[ ! -z "$HISTONE" ]]; then
		checkFILE $HISTONE
		checkCSV $HISTONE
	fi

	if [[ ! -z "$METHYL" ]]; then
		checkFILE $METHYL
		checkCSV $METHYL
	fi

	if [[ ! -z "$ATAC" ]]; then
		checkFILE $ATAC
		checkCSV $ATAC
	fi

	checkINTEGER "-p" $PROCESS

	for tool in readarray bedtools bigWigToBedGraph
	do 
		bash command -v ${tool} >/dev/null 2>&1 || { echo >&2 ${tool}" is missing (required in the PATH)"; toolMissing=1;}
	done

	if [[ "$toolMissing" == 1 ]]; then
		exit 1
	fi

	## Extract features from training intervals
	OUT=$OUTDIR/train.csv
	TMP=$OUTDIR/tmp$random
	mkdir -p $TMP
	sort -k 1,1 -k 2,2n $INTERVAL | cut -f 1-3 >$TMP/interval_srt.bed
	sort -k 1,1 -k 2,2n $INTERVAL | cut -f 4 >$TMP/summary.csv
	INTERVAL=$TMP/interval_srt.bed

	extractFeatures
	if [[ "$?" == 1 ]]; then
		exit 1
	fi

	mv $TMP/summary.csv $OUT
	rm -r $TMP

	echo "Finished extracting features for training intervals!"
fi

if [[ "$CMD" == "preprocessWG" ]]; then
	## Set defaults and checks
	if [[ -z "$OUTDIR" ]]; then
		OUTDIR=$(pwd)
	fi

	if [[ -z "$PROCESS" ]]; then
		PROCESS=1
	fi

	if [[ -z "$WINDOW_SIZE" ]]; then
		WINDOW_SIZE=500
	fi

	if [[ -z "$SLIDE_SIZE" ]]; then
		SLIDE_SIZE=100
	fi

	checkDIR $OUTDIR

	if [[ -z "$FASTA" ]]; then
		echo "ERROR: Please provide a genome sequence file in FASTA format by -f"
		echo ""
		preprocessWGHelp
	fi

	if [[ -z "$GTF" ]]; then
		echo "ERROR: Please provide an annotation GTF file by -g"
		echo ""
		preprocessWGHelp
	else
		checkGTF $GTF
	fi

	if [[ -z "$CHROMINFO" ]]; then
		echo "ERROR: Please provide a chromosome size info file by -c"
		echo ""
		preprocessWGHelp
	fi
	checkFILE $FASTA
	checkFILE $GTF
	checkFILE $CHROMINFO

	if [[ -z "$HISTONE" && -z "$METHYL" && -z "$ATAC" ]]; then
		echo "ERROR: At least one of -H, -M, and -A should be specified"
		echo ""
		preprocessWGHelp
	fi
	if [[ ! -z "$HISTONE" ]]; then
		checkFILE $HISTONE
		checkCSV $HISTONE
	fi

	if [[ ! -z "$METHYL" ]]; then
		checkFILE $METHYL
		checkCSV $METHYL
	fi

	if [[ ! -z "$ATAC" ]]; then
		checkFILE $ATAC
		checkCSV $ATAC
	fi

	checkINTEGER "-p" $PROCESS
	checkINTEGER "-w" $WINDOW_SIZE
	checkINTEGER "-s" $SLIDE_SIZE

	for tool in readarray bedtools bigWigToBedGraph
	do 
		bash command -v ${tool} >/dev/null 2>&1 || { echo >&2 ${tool}" is missing (required in the PATH)"; toolMissing=1;}
	done
	if [[ "$toolMissing" == 1 ]]; then
		exit 1
	fi

	## Extract features from whole-genome windows
	OUT=$OUTDIR/wg.csv
	TMP=$OUTDIR/tmp$random
	mkdir -p $TMP
	touch $TMP/summary.csv

	echo "Generating genome-wide windows with a window size of "$WINDOW_SIZE" and a slide size of "$SLIDE_SIZE
	bedtools makewindows -g $CHROMINFO -w $WINDOW_SIZE -s $SLIDE_SIZE | sort -k 1,1 -k 2,2n >$TMP/windows.bed
	INTERVAL=$TMP/windows.bed
	echo ""

	extractFeatures

	sed 's/^,//' $TMP/summary.csv >$OUT
	mv $TMP/windows.bed $OUTDIR
	rm -r $TMP
	echo "Finished extracting features for whole-genome windows!"
fi

if [[ "$CMD" == "train" ]]; then
	## Set defaults and checks
	if [[ -z "$OUTDIR" ]]; then
		OUTDIR=$(pwd)
	fi

	if [[ -z "$PARAMS" ]]; then
		PARAMS=""
	fi

	if [[ -z "$INPUT" ]]; then
		echo "ERROR: Please provide the input CSV file provided by preprocessTrain"
		echo ""
		trainHelp
	else
		checkFILE $INPUT
		checkCSV $INPUT
	fi

	if [[ (! -z "$n_iter" && -z "$TUNING") || (! -z "$n_jobs" && -z "$TUNING") ]]; then
		echo "ERROR: -t is required when --n_iter or --n_jobs is specified"
		exit 1
	fi

	if [[ -z "$n_iter" ]]; then
		n_iter=100
	fi

	if [[ -z "$n_jobs" ]]; then
		n_jobs=1
	fi

	checkDIR $OUTDIR
	checkINTEGER "--n_iter" $n_iter
	checkINTEGER "--n_jobs" $n_jobs

	for tool in python ledem_train.py
	do 
		command -v ${tool} >/dev/null 2>&1 || { echo >&2 ${tool}" is missing (required in the PATH)"; toolMissing=1;}
	done
	if [[ "$toolMissing" == 1 ]]; then
		exit 1
	fi

	python -c "import argparse; import textwrap; import numpy; import sklearn; import xgboost; import joblib"
	if [[ $? -eq 1 ]]; then
		echo "ERROR: Please install missing Python package(s)"
		exit 1
	fi

	## Training
	echo "Begins training..."
	echo ""
	if [[ -z "$TUNING" ]]; then
		ledem_train.py -i $INPUT -x "$PARAMS" -o $OUTDIR
	else
		ledem_train.py -i $INPUT -x "$PARAMS" -t -n_iter $n_iter -n_jobs $n_jobs -o $OUTDIR
	fi
	echo ""
	echo "Finished training an XGBoost model for training intervals!"
fi

if [[ "$CMD" == "predict" ]]; then
	## Set defaults and checks
	if [[ -z "$OUTDIR" ]]; then
		OUTDIR=$(pwd)
	fi

	if [[ -z "$DISTANCE" ]]; then
		DISTANCE=100
	fi

	if [[ -z "$INPUT" ]]; then
		echo "ERROR: Please provide the input CSV file provided by preprocessWG"
		echo ""
		predictHelp
	else
		checkFILE $INPUT
		checkCSV $INPUT
	fi

	if [[ -z "$BED" ]]; then
		echo "ERROR: Please provide the windows in BED format provided by preprocessWG"
		echo ""
		predictHelp
	else
		checkFILE $BED
		checkBED $BED
	fi

	if [[ -z "$MODEL" ]]; then
		echo "ERROR: Please provide the model generated by <train>"
		echo ""
		predictHelp
	else
		checkFILE $MODEL
	fi

	if [[ -z "$SCALER" ]]; then
		echo "ERROR: Please provide the scaler generated by <train>"
		echo ""
		predictHelp
	else
		checkFILE $SCALER
	fi

	checkDIR $OUTDIR
	checkINTEGER '-d' $DISTANCE

	for tool in python ledem_predict.py
	do 
		command -v ${tool} >/dev/null 2>&1 || { echo >&2 ${tool}" is missing (required in the PATH)"; toolMissing=1;}
	done
	if [[ "$toolMissing" == 1 ]]; then
		exit 1
	fi

	python -c "import argparse; import textwrap; import numpy; import sklearn; import xgboost; import joblib"
	if [[ $? -eq 1 ]]; then
		echo "ERROR: Please install missing Python package(s)"
		exit 1
	fi

	## Prediction
	if [[ -z "$PARAMS" ]]; then
		PARAMS=""
	fi
	echo "Begins whole-genome prediction..."
	ledem_predict.py -i $INPUT -m $MODEL -s $SCALER -o $OUTDIR
	paste $BED $OUTDIR/scores0.txt | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5}' >$OUTDIR/scores.bed
	rm $OUTDIR/scores0.txt
	awk '$5 >= 0.5' $OUTDIR/scores.bed >$OUTDIR/enhancers0.bed # windows with score >= 0.5 are output as enhancers

	bedtools merge -d $DISTANCE -i $OUTDIR/enhancers0.bed >$OUTDIR/enhancers.bed
	rm $OUTDIR/enhancers0.bed
	echo "Finished whole-genome prediction!"
fi
