from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import sys

IN_FILE_ADDR = sys.argv[1]
OUT_FILE_ADDR = sys.argv[2]

# sentence length should be in (5,200)
min_src_ntokens = 5
max_src_ntokens = 200
min_nsents = 3
max_nsents = 100

f = open(IN_FILE_ADDR, "r")
lines = f.readlines()
lines = [i.split('\n')[0] for i in lines]

res = []
f = open(OUT_FILE_ADDR, "w")

# nltk
for line in tqdm(lines):
    sents = sent_tokenize(line)
    idxs = [i for i, s in enumerate(sents) if (len(s) > min_src_ntokens)]
    sents = [sents[i][:max_src_ntokens] for i in idxs]
    sents = sents[:max_nsents]

    processed_sent = " ".join(["%s%s" % ("<cls>", sent) for sent in sents])
    f.write("%s\n" % processed_sent)


