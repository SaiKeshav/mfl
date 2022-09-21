import pickle
from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE
from tqdm import tqdm
import os
import json
import ipdb
import glob
import ast
import random

from absl import flags
from absl import app
from concurrent import futures
from itertools import repeat

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_string('trie_type', '', 'fact/entity/predicate/dict separate multiple by _')


def main(unused_argv):
  triples = [json.loads(s.strip()) for s in open('../data/WikiData_facts.jsonl','r').readlines()]

  predicate_d, object_d, subject_d, entity_d = {}, {}, {}, {}

  genre_model = mGENRE.from_pretrained('../models/fairseq_multilingual_entity_disambiguation/', checkpoint_file='model.pt', bpe='sentencepiece', layernorm_embedding=True, sentencepiece_model="spm_256000.model")
  # genre_model = mGENRE.from_pretrained('../models/mbart.cc25.v2/', checkpoint_file='model.pt', bpe='sentencepiece', layernorm_embedding=True, sentencepiece_model="sentence.bpe.model")

  total_triples = 0
  entity_titles = []
  ptriples = []

  no_en_desc = 0
  print('Collecting all facts labels...')
  for triple_i, triple in tqdm(enumerate(triples)):
    if FLAGS.debug and triple_i > 1000:
      break

    canon_triple = triple["fact"]
    label_d = triple["fact_label"]

    if 'en' not in label_d:
      no_en_desc += 1
      continue

    subj_qid, predicate_id, obj_qid = canon_triple.split(';')
    fields = label_d['en'].split(';')    
    subj_title = fields[0].strip()
    predicate = fields[1].strip()
    obj_title = ';'.join(fields[2:]).strip()

    if predicate_id not in predicate_d:
      predicate_d[predicate] = predicate_id
    if obj_qid not in entity_d:
      entity_d[obj_title] = obj_qid
    if subj_qid not in entity_d:
      entity_d[subj_title] = subj_qid
    
    entity_titles.append(subj_title)
    entity_titles.append(obj_title)

    ptriple = subj_title+" ; "+predicate+" ; "+obj_title
    ptriples.append(ptriple)
    total_triples += 1

  entity_titles = list(set(entity_titles))

  FLAGS.trie_type = FLAGS.trie_type.split('_')

  print('Computing the BPE Encodings...')
  unk = 0
  if 'fact' in FLAGS.trie_type: 
    encoded_triples = []
    for ptriple in tqdm(ptriples):
      encoded_triple = [2]+genre_model.encode(ptriple).tolist()[1:]
      encoded_triple = [t if t < 256001 else 3 for t in encoded_triple]
      if 3 in encoded_triple:
        unk += 1
      encoded_triples.append(encoded_triple)
    encoded_triples.append([2]+genre_model.encode('None ; None ; None').tolist()[1:])    
    t = MarisaTrie(encoded_triples)
    if FLAGS.debug:
      t = MarisaTrie([[2]+genre_model.encode('None ; None ; None').tolist()[1:]])
      pickle.dump(t, open('../data/fact_trie_debug.pkl','wb'))
    else:
      pickle.dump(t, open('../data/fact_trie.pkl','wb'))

  if 'entity' in FLAGS.trie_type:
    encoded_entities = []
    for entity_title in tqdm(entity_titles):
      encoded_entity = [2]+genre_model.encode(ptriple+" >> en").tolist()[1:]
      encoded_entity = [t if t < 256001 else 3 for t in encoded_entity]
      if 3 in encoded_entity:
        unk += 1
      encoded_entities.append(encoded_entity)
    encoded_entities.append([2]+genre_model.encode('None >> en').tolist()[1:])    
    t = MarisaTrie(encoded_entities)
    pickle.dump(t, open('../data/entity_trie.pkl','wb'))

  if 'predicate' in FLAGS.trie_type:
    encoded_predicates = []
    for predicate in tqdm(predicate_d):
      encoded_predicate = [2]+genre_model.encode(predicate+" >> en").tolist()[1:]
      encoded_predicate = [t if t < 256001 else 3 for t in encoded_predicate]
      if 3 in encoded_predicate:
        unk += 1
      encoded_predicates.append(encoded_predicate)
    encoded_predicates.append([2]+genre_model.encode('None >> en').tolist()[1:])    
    t = MarisaTrie(encoded_predicates)
    pickle.dump(t, open('../data/predicate_trie.pkl','wb'))

  if 'dict' in FLAGS.trie_type and not FLAGS.debug:
    pickle.dump(predicate_d, open('../data/predicate_id.pkl', 'wb'))
    pickle.dump(entity_d, open('../data/entity_id.pkl', 'wb'))

  print("Total number of facts = ", total_triples)
  print("Num facts with no English descriptions = ", no_en_desc)
  print("Total number of encodings with unk in them = ", unk)

if __name__ == '__main__':
  app.run(main)
