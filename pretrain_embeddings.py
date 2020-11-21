import fasttext
import sys
file = sys.argv[1]
model = fasttext.train_unsupervised(file, model='skipgram', ws=5, dim=1024, neg=10, minCount=0)
model.save_model(file + "_emb.bin")

