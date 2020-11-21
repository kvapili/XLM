head -100000 data/processed/cs-de_26M/train.de | ./translate.sh dumped/unsupMT_decs/mono_bt_ae_26M_synthpretr_2400x8/best-valid_de-cs_mt_bleu.pth data/processed/cs-de_26M/100k.de.translated.best.cs de cs
head -100000 data/processed/cs-de_26M/train.de | ./translate.sh dumped/unsupMT_decs/mono_bt_ae_26M_2400x8/best-valid_de-cs_mt_bleu.pth data/processed/cs-de_26M/100k.de.translated.cs de cs
head -100000 data/processed/cs-de_26M/train.cs | ./translate.sh dumped/unsupMT_decs/mono_bt_ae_26M_2400x8/best-valid_de-cs_mt_bleu.pth data/processed/cs-de_26M/100k.cs.translated.de cs de
head -100000 data/processed/en-fr/train.en | ./translate.sh dumped/unsupMT_enfr/elw1r68nlb/best-valid_en-fr_mt_bleu.pth data/processed/en-fr/100k.en.translated.fr en fr
head -100000 data/processed/en-fr/train.fr | ./translate.sh dumped/unsupMT_enfr/elw1r68nlb/best-valid_en-fr_mt_bleu.pth data/processed/en-fr/100k.fr.translated.en fr en


