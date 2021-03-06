#source /home/kvapilikova/personal_work_ms/venvs/cuda_92/bin/activate
source /home/kvapilikova/personal_work_ms/venvs/xlm/bin/activate

# Default arguments
if [[ `hostname` == *tdll* ]]; then
   BS=3450
else
   BS=1600
fi

ACC=2
SUFFIX=""
MODEL=""
#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --accumulate)
    ACC="$2"; shift 2;;
  --ngpu)
    export NGPU="$2"; shift 2;;
  --src)
    L1="$2"; shift 2;;
  --tgt)
    L2="$2"; shift 2;;
  --bs)
    BS="$2"; shift 2;;
  --suffix)
    SUFFIX="$2"; shift 2;;
  --model)
    MODEL="$2"; shift 2;;
  --name)
    NAME="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


if [[ $L1 > $L2 ]]; then
    SRC=$L2
    TGT=$L1
else
    SRC=$L1
    TGT=$L2
fi

EXP_NAME="mlia"
EXP_ID="${SRC}-${TGT}_mt_bt_${NAME}_${ACC}x${BS}x${NGPU}"

TMP=.`basename "$0"`.$RANDOM 
cp "$0" $TMP

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
\
`## main parameters` \
--exp_name "$EXP_NAME"                                       `# experiment name` \
--exp_id "$EXP_ID" \
--dump_path ./dumped/                                         `# where to store the experiment` \
--reload_model "$MODEL,$MODEL"           \
`#--reload_checkpoint 'dumped/unsupMT_dehsb/csde_hsbde_dehsb_finetune_dehsb_para_conr_8x1400x1/checkpoint.pth'` \
\
`## data location / training objective` \
--data_path ./data/processed/mlia/${SRC}-${TGT}_${SRC}${TGT}${SUFFIX}                        `# data location` \
--lgs "$SRC-$TGT"                                                 `# considered languages` \
--bt_steps "$SRC-$TGT-$SRC,$TGT-$SRC-$TGT"   \
--mt_steps "$SRC-$TGT,$TGT-$SRC" \
\
`## transformer parameters` \
--encoder_only false                                          `# use a decoder for MT` \
--emb_dim 1024                                                `# embeddings / model dimension` \
--n_layers 6                                                  `# number of layers` \
--n_heads 8                                                   `# number of heads` \
--dropout 0.1                                                 `# dropout` \
--attention_dropout 0.1                                       `# attention dropout` \
--gelu_activation true                                        `# GELU instead of ReLU` \
\
`## optimization` \
--tokens_per_batch $BS                                       `# use batches with a fixed number of words` \
--batch_size 30                                               `# batch size (for back-translation)` \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  `# optimizer` \
--epoch_size 60000                                           `# number of sentences per epoch` \
--eval_bleu true                                              `# also evaluate the BLEU score` \
--stopping_criterion "valid_${SRC}-${TGT}_mt_bleu,10"                 `# validation metric (when to save the best model)` \
--validation_metrics "valid_${SRC}-${TGT}_mt_bleu,valid_${TGT}-${SRC}_mt_bleu"                    `# end experiment if stopping criterion does not improve` \
--fp16 True \
--amp 1 \
--accumulate_gradients $ACC #\
#--beam_size 4 

mv "$TMP" ./dumped/"$EXP_NAME"/"$EXP_ID" 
