source /home/kvapilikova/personal_work_ms/venvs/xlm/bin/activate

if [[ `hostname` == *tdll* ]]; then
   BS=256
else
   BS=180
fi

python translate.py \
\
`## main parameters` \
--exp_name translate                                       `# experiment name` \
--dump_path ./dumped/                                         `# where to store the experiment` \
--model_path $1 \
--output_path $2 \
--src_lang $3 \
--tgt_lang $4 \
--batch_size $BS

