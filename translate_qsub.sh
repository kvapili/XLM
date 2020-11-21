MODELDIR="dumped/unsupMT_dehsb/dehsb_synth_new_pretr_attempt2_bt_mt_8x1600x1"
if [[ $1 == -1 ]]; then
cat data/mono/hsb_wmt20_3way/all.hsb.tok | tools/fastBPE/fast applybpe_stream data/processed/cs-de-hsb_wmt20/codes data/processed/cs-de-hsb_wmt20/vocab.cs-de-hsb | ./translate.sh dumped/unsupMT_dehsb/dehsb_synth_new_pretr_bt_mt_8x3450x1/best-valid_hsb-de_mt_bleu.pth dumped/unsupMT_dehsb/dehsb_synth_new_pretr_bt_mt_8x3450x1/all.hsb.translated.de.tok hsb de 
else
head -n $1 data/mono/de_wmt20_3way/all.de.tok | tail -n $2 | tools/fastBPE/fast applybpe_stream data/processed/cs-de-hsb_wmt20/codes data/processed/cs-de-hsb_wmt20/vocab.cs-de-hsb | ./translate.sh $MODELDIR/best-valid_de-hsb_mt_bleu.pth $MODELDIR/$1_$2.de.translated.hsb.tok de hsb 
fi
#tail -n 500000 data/mono/de_wmt20_3way/all.de.tok | tools/fastBPE/fast applybpe_stream data/processed/de-hsb_wmt20/codes data/processed/de-hsb_wmt20/vocab.de-hsb | ./translate.sh dumped/unsupMT_dehsb/dehsb_bt_ae_3550x4/best-valid_hsb-de_mt_bleu.pth data/mono/de_wmt20_3way/last500k.de.translated.hsb.tok de hsb 
#tail -n +1000001 data/mono/de_wmt20_3way/all.de.tok | tools/fastBPE/fast applybpe_stream data/processed/de-hsb_wmt20/codes data/processed/de-hsb_wmt20/vocab.de-hsb | ./translate.sh dumped/unsupMT_dehsb/dehsb_bt_ae_3550x4/best-valid_hsb-de_mt_bleu.pth data/mono/de_wmt20_3way/rest.de.translated.hsb.tok de hsb 
#cat data/mono/hsb_wmt20_3way/all.hsb.tok | tools/fastBPE/fast applybpe_stream data/processed/de-hsb_wmt20/codes data/processed/de-hsb_wmt20/vocab.de-hsb | ./translate.sh dumped/unsupMT_dehsb/dehsb_bt_ae_3550x4/best-valid_hsb-de_mt_bleu.pth data/mono/hsb_wmt20_3way/all.hsb.translated.de.tok hsb de
#head data/mono/de_wmt20_pbmt/all.de.tok | ./translate.sh dumped/unsupMT_dehsb/dehsb_synth3_bt_mt_3450x1/best-valid_hsb-de_mt_bleu.pth dumped/unsupMT_dehsb/dehsb_synth3_bt_mt_3450x1/pokus.hsb de hsb
