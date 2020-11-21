# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e


N_THREADS=16    # number of threads in data preprocessing

BEAM_SIZE=1
#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --model)
    MODEL="$2"; shift 2;;
  --bpe)
    BPE_CODES="$2"; shift 2;;
  --vocab)
    VOCAB="$2"; shift 2;;
  --input)
    TEST_SET="$2"; shift 2;;
  --ref)
    REF="$2"; shift 2;;
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --lc_no_acc)
    LC_NO_ACC="$2"; shift 2;;
  --beam_size)
    BEAM_SIZE="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


#
# Check parameters
#
if [ "$MODEL" == "" ]; then echo "--model not provided"; exit; fi
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$TEST_SET" == "" ]; then echo "--input not provided"; exit; fi


if [ "$VOCAB" == "" ]; then 
    BPE_CODES=`echo $MODEL | sed 's/.pth/.fcodes/'`
    VOCAB=`echo $MODEL | sed 's/.pth/.fvocab/'`
fi
    

#
# Initialize tools and data paths
#
TEST_SET_BASE=`basename $TEST_SET | sed 's/.sgm//'`
TEST_SET_TOK=$PWD/$TEST_SET_BASE.tok
TEST_SET_BPE=$PWD/$TEST_SET_BASE.bpe
TEST_SET_OUT_TOK=`dirname $MODEL`/$TEST_SET_BASE.translated.$TGT.tok
TEST_SET_OUT=`dirname $MODEL`/$TEST_SET_BASE.translated.$TGT
BLEU=$TEST_SET_OUT.bleu

REF_BASE=`basename $REF`
REF_TOK=$REF.tok
# main paths
TOOLS_PATH=$PWD/tools

# moses
MOSES=$TOOLS_PATH/mosesdecoder
MULTIBLEU=$MOSES/scripts/generic/multi-bleu.perl
MTEVAL=$MOSES/scripts/generic/mteval-v14.pl
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
DETOKENIZER=$MOSES/scripts/tokenizer/detokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

LOWERCASE_AND_REMOVE_ACCENT="python $TOOLS_PATH/lowercase_and_remove_accent.py"


LC_NO_ACC=`echo $LC_NO_ACC | tr '[:upper:]' '[:lower:]'`
echo $LC_NO_ACC

if [ "$LC_NO_ACC" == "true" ]; then
  PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |             $LOWERCASE_AND_REMOVE_ACCENT | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
else
  PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
fi
  
PREPROCESSING_TGT="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

# check valid and test files are here
if ! [[ -f "$TEST_SET" ]]; then echo "$TEST_SET is not found!"; exit; fi

echo "Tokenizing valid and test data..."
if [ `echo "$TEST_SET" | rev | cut -c 1-3 | rev` == sgm ]; then
eval "$INPUT_FROM_SGM < $TEST_SET | $PREPROCESSING > $TEST_SET_TOK"
else
    eval "cat $TEST_SET | $PREPROCESSING > $TEST_SET_TOK"
fi

#echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $TEST_SET_BPE  $TEST_SET_TOK  $BPE_CODES $SRC_VOCAB

echo "cat $TEST_SET_BPE | ./translate.sh --model_path $MODEL --output_path $TEST_SET_OUT_TOK --src_lang $SRC --tgt_lang $TGT"
cat $TEST_SET_BPE | python ./translate.py --exp_name translate --model_path $MODEL --output_path $TEST_SET_OUT_TOK --src_lang $SRC --tgt_lang $TGT  --beam_size $BEAM_SIZE > /dev/null
sed -r 's/(@@ )|(@@ ?$)//g' $TEST_SET_OUT_TOK

$DETOKENIZER -l $TGT <  $TEST_SET_OUT_TOK > $TEST_SET_OUT

if [ `echo "$REF" | rev | cut -c 1-3 | rev` == sgm ]; then
eval "$INPUT_FROM_SGM < $REF | $PREPROCESSING_TGT > $REF_TOK"
else
eval "cat $REF | $PREPROCESSING_TGT > $REF_TOK"
fi

if [ $REF != "" ]; then
$MULTIBLEU $REF_TOK < $TEST_SET_OUT_TOK |& tee $BLEU
#$MTEVAL $REF < $TEST_SET_OUT
fi 

rm $TEST_SET_OUT_TOK $TEST_SET_TOK  
