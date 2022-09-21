import os
import ipdb
import sys
import json

data_dir = '../data/mokbc_testsets/'
# for fil in ['test_en.txt', 'test_hi.txt', 'test_te.txt']:
for lang in ['en', 'hi', 'te']:
    full_fil = data_dir+'test_'+lang+'.txt'
    triples = open(full_fil, encoding='utf-8').readlines()
    ofp = open(data_dir+'test_'+lang+'.jsonl','w')
    for tid, triple in enumerate(triples):
        triple = ' '.join([t.strip() for t in triple.strip().split('\t')])
        ofp.write(json.dumps({'sent_id': tid, 'language': lang, "sentence": triple, "facts": []}, ensure_ascii=False)+'\n')
ofp.close()
