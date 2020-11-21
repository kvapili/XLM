import argparse
from collections import defaultdict, Counter
from shlex import quote
import subprocess
import numpy as np
import os

def length_normalize(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0]
    matrix /= norms[:, np.newaxis]

parser = argparse.ArgumentParser()
parser.add_argument('--emb_path', type=str,
                            help='Path to VECMAP embeddings')
parser.add_argument('--in_path', type=str, 
                            help='Path to data to embed')
parser.add_argument('--out_path', type=str, 
                            help='Path to output file')
parser.add_argument('--truecase_model', type=str, 
                            help='Path to truecase model')
parser.add_argument('--bpe_codes', type=str, required=False,
                            help='Path to BPE codes')
parser.add_argument('--lang', type=str, 
                            help='Laanguage')
parser.add_argument('--threads', type=int, default=1, 
                            help='Number of threads')
parser.add_argument('--normalize', action='store_true',
                            help='Normalize word embeddings')

args = parser.parse_args()

MOSES='/home/kvapilikova/personal_work_ms/workspace/mosesdecoder/'
FASTBPE='/lnet/tspec/work/people/kvapilikova/XLM/tools/fastBPE/fast'

def bash(command):
    subprocess.run(['bash', '-c', command])

#with open(args.emb_path, 'r', encoding='utf-8') as f:
#    header = f.readline().split()
#    n_words = int(header[0])
#    dim = int(header[1])

#bash('tail -n ' + n_words + quote(args.emb_path) + ' | cut -d" "  -f1' + quote(args.emb_path) + ' > ' + quote(args.emb_path + '.w') )
#bash('tail -n ' + n_words + quote(args.emb_path) + ' | cut -d" "  -f2:' + str(dim + 1) + ' '  + quote(args.emb_path + '.vec')) 
tokenized_path = args.in_path + '.tok'
truecased_path = args.in_path + '.true'
segmented_path = args.in_path + '.bpe'

if not os.path.exists(tokenized_path):
    if args.lang == 'zh':
        bash("python -m jieba -d' ' < " + quote(args.in_path) + ' > ' + quote(tokenized_path))
    else:
        bash(quote(MOSES + '/scripts/tokenizer/tokenizer.perl') +
             ' -l ' + quote(args.lang) + ' -threads ' + str(args.threads) +
                          ' < ' + quote(args.in_path) + ' > ' + quote(tokenized_path))
    print('Tokenization done')
else:
    print('Tokenized text exists.')


if not os.path.exists(truecased_path) and args.truecase_model:
    bash(quote(MOSES + '/scripts/recaser/truecase.perl') +
            ' --model ' + quote(args.truecase_model) +
              ' < ' + quote(tokenized_path ) +
              ' > ' + quote(truecased_path))
    print('Truecase done')


if not os.path.exists(segmented_path) and args.bpe_codes:
    bash(quote(FASTBPE) + ' applybpe ' + quote(segmented_path) + ' ' + quote(tokenized_path) + ' ' + quote(args.bpe_codes))
    print('BPE applied.')

if args.bpe_codes:
    preprocessed_path = segmented_path
elif args.truecase_model:
    preprocessed_path = truecased_path 
else:
    preprocessed_path = tokenized_path


with open(args.emb_path, 'r', encoding='utf-8', errors='ignore') as f:
    header = f.readline().split()
    n_words = int(header[0])
    dim = int(header[1])
    emb_dict = {}
    for i in range(n_words):
        if i % 5000 == 0:
            print('%d words processed.' % i, end ='\r')
        line = f.readline().split()
        word = line[0]
        if len(word.split()) > 1:
            continue
        vector = np.array(line[1:], dtype=np.float32)
        if args.normalize:
            vector /= np.sqrt(np.sum(vector ** 2))
        emb_dict[word] = vector

with open(preprocessed_path) as f:
    unk_words = 0
    sentences = [sent.split() for sent in f.readlines()]
    ##tf = Counter([word for sentence in sentences for word in sentence])
    ##idf = {word: sum([1 for sentence in sentences if word in sentence]) for word in emb_dict}
    mode = "average"
    if mode == "weighted":
        df = Counter()
        tf = Counter()
        for sentence in sentences:
            for word in set(sentence):
                df[word] += 1
            for word in sentence:
                tf[word] += 1
    else:
        tf = defaultdict(lambda: 1)
        df = defaultdict(lambda: 1)

    sentence_embeddings = np.zeros([len(sentences), dim], dtype = np.float32)
    print(dim, len(sentences))
    for i, sentence in enumerate(sentences):
        words_per_sentence = 0
        weighting_sum = 0
        sent_embedding = np.zeros(dim)
        for word in sentence:
            if word not in emb_dict:
 #               print('unk')
                unk_words += 1
            else:
                word_embedding = emb_dict[word]
                words_per_sentence += 1
                weight = tf[word]/df[word]
                sent_embedding = sent_embedding + weight * word_embedding
                weighting_sum += weight
#                print('Word: %s, weight: 1/%d' % (word, df[word]))
        if words_per_sentence == 0:
            print("Warning: All <UNK>: " + ' '.join(sentence))
            for word in sentence:
                for c in reversed(range(len(word))):
                    if word[:c] in emb_dict:
                        word_embedding = emb_dict[word[:c]]
                        words_per_sentence += 1
                        sent_embedding = sent_embedding + word_embedding
                        weighting_sum += 1
                        print('Found ' + word[:c])
                        break
        if words_per_sentence == 0:
            sent_embedding = np.random.rand(dim)
            words_per_sentence = 1
            weighting_sum += 1
            print("random init")
        sentence_embeddings[i] = sent_embedding / weighting_sum

    print('Total %d unknown words.' % unk_words)
    sentence_embeddings.tofile(args.out_path)




