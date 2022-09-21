import json
import os
import re
import csv
import copy
import ast
import sys
import math
import ipdb
import pickle
import random
import numpy as np
from tqdm import tqdm
import faiss
import torch

from concurrent import futures
from itertools import repeat
from absl import app
from absl import flags
from absl import logging
from torch import nn
from torch.nn import functional
from torch.utils import data
from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation, SentencesDataset
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

import utils

"""Training/Testing Dual Encoder models and generating data for Ranker models
"""


FLAGS = flags.FLAGS

# General flags
flags.DEFINE_string('de_mode', 'train.test', 'Dual Encoder modes')
flags.DEFINE_string('ranker_mode', 'test.dev.train', 'Generate rank data only on these modes')
flags.DEFINE_boolean('debug', False, 'Run in debug mode for quick testing')

# Flags for building/saving triple index/rank data
flags.DEFINE_boolean('build_index', False, 'Build the triple index')
flags.DEFINE_boolean('dont_save_index', False, 'Dont save the triple index')
flags.DEFINE_boolean('dont_save_rank_data', False, 'Dont save the rank data')
flags.DEFINE_boolean('append_rank_data', False, 'Append rank data to existing file')

# Languages for DE/Ranker models
flags.DEFINE_string('input_languages', 'en', 'list of input languages, separated by .')
flags.DEFINE_string('retrieval_languages', 'en', 'list of fact languages, separated by .')
flags.DEFINE_string('fact_set', 'en', 'Fact set on which retrieval is done. il/en/en_il/all')
flags.DEFINE_string('ranker_languages', 'en', 'list of cross languages, separated by .')

# Paths for model/data
flags.DEFINE_string('data_dir', '../data', 'data dir')
flags.DEFINE_string('model_dir', '../models', 'model dir')

flags.DEFINE_string('triples_path', None, 'data dir')
flags.DEFINE_string('train_path', None, 'data dir')
flags.DEFINE_string('test_path', None, 'data dir')

flags.DEFINE_string('model_save_path', 'dummy', 'Path to saved model')
flags.DEFINE_string('model_load_path', '', 'Load saved model to continue training')

# Optimizaton hyper-parameters
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('epochs', 5, 'epochs')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')

# Misc. flags
flags.DEFINE_boolean('all_langs_test', False, 'Consider only test examples which have all language representations')
flags.DEFINE_string('accumulate_scores', 'max', 'sum/max')
flags.DEFINE_integer('topk', 5, 'Top-k facts from Dual Encoder')

indiclink_languages = ['en','hi','te','ta','ur','gu','as']

def read_jsonl(file_path, triples_d, test):
  train_examples = {'en': [], 'hi': [], 'te': [], 'ta': [], 'ur': [], 'gu': [], 'as': []}
  dev_examples = {'en': [], 'hi': [], 'te': [], 'ta': [], 'ur': [], 'gu': [], 'as': []}
  none_d = {'en': 'None ; None ; None'}

  fact_descs_not_found, total_facts = 0, 0
  for line in open(file_path,'r').readlines():
    if not test and random.randint(0,9) == 0:
      examples = dev_examples
    else:
      examples = train_examples

    jobj = json.loads(line.strip())
    valid_fact_found = False
    pro_facts = []
    not_found = False

    for fact in jobj['facts']:
      relation = fact['fact'].split(';')[1]
      if 'P0' in relation or 'None' in relation:
        continue
      valid_fact_found = True
      total_facts += 1

      if test:
        if fact['fact'] in triples_d:
          fact_desc = triples_d[fact['fact']]
        else:
          fact_desc = none_d    
      else:
        fact_desc = {'en': fact['fact']}

      examples[jobj['language']].append([jobj['sent_id'], 
                                              jobj['sentence'],
                                              fact['fact'],
                                              fact_desc])
    if not valid_fact_found:
      if test:
        none_desc = 'Q0;P0;Q0'
      else:
        none_desc = 'None ; None ; None'

      examples[jobj['language']].append([jobj['sent_id'], 
                                              jobj['sentence'],
                                              none_desc,
                                              none_d])

  if test:
    return train_examples
  else:
    return train_examples, dev_examples

def process_data(data, fact_language):
  '''Convert data into (sentence, fact, fields)
  '''
  dataset = []
  for example in data:    
    sentence_id, sentence, canon_fact, fact_lang_d = example
    if fact_language not in fact_lang_d:
      continue

    # Invert s, r, o order for Urdu examples
    if 'ur' in fact_lang_d:
      fact_lang_d['ur'] = ';'.join(fact_lang_d['ur'].split(';')[::-1])

    fact = fact_lang_d[fact_language]

    fields = []
    fields.extend([r.strip() for r in fact.split(';')])
    fields.extend(canon_fact.split(';'))
    fields.append(sentence_id)

    dataset.append([sentence, fact, fields])

  return dataset

def process_train_examples(fact_language, train_data):
  '''Read train examples
  '''
  dataset = process_data(train_data, fact_language)

  examples = []
  for datum in dataset:
    sentence, fact, fields = datum
    examples.append(InputExample(texts=[sentence, fact], label=1.0))

  return examples

def process_eval_examples(data_d):
  '''Read eval data
  '''
  dataset = []
  for language in data_d:
    dataset.extend(process_data(data_d[language], 'en'))

  queries, relevant, corpus = {}, {}, {}
  sro_d, q2i_d = {}, {}

  for i, datum in enumerate(dataset):
    if FLAGS.debug and i > 10:
      break
    sentence, fact, fields = datum
    sro_d[fact] = fields

    qindex = f'Q{i}'
    if sentence not in q2i_d:
      q2i_d[sentence] = qindex
      queries[qindex] = sentence
    else:
      qindex = q2i_d[sentence]

    rindex = f'R{i}'
    corpus[rindex] = fact
    if qindex not in relevant:
      relevant[qindex] = set([rindex])
    else:
      relevant[qindex].add(rindex)

  return queries, corpus, relevant, sro_d


def retrieve_facts(examples,
                  model,
                  index,
                  triple_sro_d,
                  canon_desc_d,
                  topk,
                  triples,
                  input_language,
                  mode):
  '''Retrieve facts from Dual Encoder model and generate ranker data
  '''
  queries, corpus, relevant, sroD = process_eval_examples({input_language: examples})
                                                              
  ranker_data_d = {'fact': [], 'sid': []}
  # fact_hitsk, entity_hitsk, relation_hitsk = [0]*topk, [0]*topk, [0]*topk
  # subj_hitsk, obj_hitsk = [0]*topk, [0]*topk
  # sum_rank = 0
  gold_ids, all_gold_facts, all_gold_en_facts, all_retrieved_facts = [], [], [], []
  print('Input Language: ', input_language, ', Mode: ', mode)
  print('Number of test examples = %d'%(len(queries)))

  items = list(queries.items())
  indices = [item[0] for item in items]
  queries = [item[1] for item in items]

  ## Remove device='cpu' for GPU testing
  # model = model.cpu()
  embs = model.encode(queries,
                      batch_size=2048,
                      convert_to_tensor=True,
                      normalize_embeddings=True)
                      # device='cpu')

  ## Uncomment below two lines for GPU testing
  res = faiss.StandardGpuResources()
  index = faiss.index_cpu_to_gpu(res, 0, index)
  scores, result = index.search(embs.cpu().numpy(), k=FLAGS.topk)

  total, correct_e1, correct_r = 0, 0, 0
  relation_d = {}
  for result_index, (score_item, result_item) in enumerate(zip(scores, result)):
    query_key = indices[result_index]
    query = queries[result_index].strip('\n')
    gold_fact_keys = relevant[query_key]
    gold_facts = []
    for gfk in gold_fact_keys:
      fact = corpus[gfk]
      fields = sroD[fact]
      canon_fact = ';'.join(fields[3:6])
      gold_facts.append(canon_fact)
      relation_d[fields[4]] = fields[1]

    ranker_data_d['sid'].append([fields[-1], fields[-1]])

    retrieved_facts, retrieved_triples, rankl_facts = [], [], []
    result_d = {}

    for topi, (score, corpus_id) in enumerate(zip(score_item, result_item)):
      if topi == topk:
        break

      fact = triples[corpus_id]
      triple_fields = triple_sro_d[fact]
      canon_fact = ';'.join(triple_fields[3:6])

      if canon_fact not in result_d:
        result_d[canon_fact] = [triple_fields, score]
      elif FLAGS.accumulate_scores == 'sum':
        result_d[canon_fact][1] += score

    retrieved_facts = sorted(result_d.items(), key=lambda x: x[1][1], reverse=True)

    for canon_fact, (triple_fields, score) in retrieved_facts:
      lang_d = triple_fields[6]
      # lang_d = utils.add_languages(lang_d, FLAGS.input_languages, None)
      if FLAGS.ranker_languages == 'en':
        ranker_languages = set(['en'])
      elif FLAGS.ranker_languages == 'en_il':
        ranker_languages = set([input_language, 'en'])
      elif FLAGS.ranker_languages == 'all':
        ranker_languages = indiclink_languages
      else:
        print('Retrieval language option not defined!')

      rankl_str = ''
      for language in ranker_languages:
        if language in lang_d:
          rankl_str += lang_d[language]+' : '

      rankl_str = rankl_str.strip(' : ')
      rankl_facts.append(rankl_str)

    retrieved_facts = [r[0] for r in retrieved_facts]

    # found_fact, found_subj, found_obj, found_relation = False, False, False, False
    # rank = topk
    # ret_pred = 'P0'
    # ret_correct = []
    # for k, retrieved_fact in enumerate(retrieved_facts):
    #   ret_subj, ret_pred, ret_obj = retrieved_fact.split(';')
    #   none_fact = False
    #   for gold_fact in gold_facts:
    #     gold_subj, gold_pred, gold_obj = gold_fact.split(';')
    #     if 'P0' in gold_fact:
    #       none_fact = True

    #     if gold_subj == ret_subj and gold_pred == ret_pred and gold_obj == ret_obj:
    #       found_fact = True
    #       if k < rank:
    #         rank = k

    #     if gold_subj == ret_subj:
    #       found_subj = True
    #     if gold_obj == ret_obj:
    #       found_obj = True
    #     if gold_pred == ret_pred:
    #       found_relation = True

    #   if none_fact and found_relation:
    #     found_subj, found_obj, found_fact = True, True, True

    #   if found_fact:
    #     fact_hitsk[k] += 1
    #     ret_correct.append(1)
    #   else:
    #     ret_correct.append(0)

    #   if found_subj:
    #     subj_hitsk[k] += 1
    #     entity_hitsk[k] += 1
    #   if found_obj:
    #     obj_hitsk[k] += 1
    #     entity_hitsk[k] += 1
    #   if found_relation:
    #     relation_hitsk[k] += 1

    # sum_rank += rank
    # total += 1

    # gsids, goids, gpids = [], [], []
    # gold_facts_str = []
    # for gold_fact in gold_facts:
    #   gold_subject_qid, gold_relation_qid, gold_object_qid = gold_fact.split(';')
    #   gold_fact_label = canon_desc_d[gold_fact]
    #   if 'en' in gold_fact_label:
    #     gold_fact_fields = gold_fact_label['en'].split(';')
    #     subject_title = gold_fact_fields[0]
    #     relation_title = gold_fact_fields[1]
    #     object_title = ' '.join(gold_fact_fields[2:])
    #   else:
    #     subject_title = relation_title = object_title = ''

    #   gold_facts_str.append(
    #       f'{subject_title} ; {relation_title} ; {object_title} >> en')

    #   gsids.append(gold_subject_qid)
    #   goids.append(gold_object_qid)
    #   gpids.append(gold_relation_qid)

    # gold_ids.append(f'{",".join(gsids)};{",".join(gpids)};{",".join(goids)}')
    # all_gold_en_facts.append(' [SEP] '.join(gold_facts_str))
    # all_gold_facts.append(' [SEP] '.join(gold_facts))

    all_retrieved_facts.append(' [SEP] '.join(retrieved_facts))

    concat_facts = ' [SEP] ' + ' [DEP] '.join(rankl_facts)

    # query = query.replace('{ ', '')
    # query = query.replace(' }', '')
    fact_query = query + concat_facts
    # rel_query = subj_query = obj_query = fact_query = cquery

    if mode != 'test':
      for gold_fact in gold_facts:
        # try:
        #   gold_fact_desc = canon_desc_d[gold_fact]
        #   en_gold_fact_desc = gold_fact_desc['en']
        # except:
        #   en_gold_fact_desc = 'None ; None ; None' 
        #   fact_descs_not_found += 1
        ranker_data_d['fact'].append([fact_query, gold_fact])
    else:
      # try:
      #   gold_fact_desc = canon_desc_d[gold_facts[0]]
      #   en_gold_fact_desc = gold_fact_desc['en'] if 'en' in gold_fact_desc else  
      # except:
      #   en_gold_fact_desc = 'None ; None ; None'
      #   fact_descs_not_found += 1
      ranker_data_d['fact'].append([fact_query, 'None ; None ; None'])

  # if topk != 0:
  #   performance = ['%.1f'%(100*subj_hitsk[0]/total),
  #                  '%.1f'%(100*obj_hitsk[0]/total),
  #                  '%.1f'%(100*entity_hitsk[0]/(2*total)),
  #                  '%.1f'%(100*relation_hitsk[0]/total),
  #                  '%.1f'%(100*fact_hitsk[0]/total)]
  # else:
  #   performance = ['0', '0', '0', '0', '0']

  # print(performance)
  # return performance, ranker_data_d, all_gold_facts, all_gold_en_facts, all_retrieved_facts

  # print('Number of facts where description is not found = ', en_gold_fact_desc)

  return ranker_data_d, all_retrieved_facts

def load_model():
  '''Load the SentenceTransformer model
  '''
  if FLAGS.model_load_path:
    model = SentenceTransformer(FLAGS.model_load_path)
  else:
    model = SentenceTransformer('LaBSE')
  model._first_module().max_seq_length = 64
  return model

def train(model,
          train_examples_d,
          dev_examples_d):
  '''Train the Dual Encoder model
  '''

  # train_objectives = []
  # train_loss = losses.MultipleNegativesRankingLoss(model)

  # il_set = set(input_languages)
  # all_train_examples, all_train_sro_d = {il: [] for il in il_set
  #                                       }, {il: {} for il in il_set}
  all_train_examples = []
  for il in FLAGS.input_languages:
    for fl in FLAGS.retrieval_languages:
      train_examples = process_train_examples(fl, train_examples_d[il])
      all_train_examples.extend(train_examples)
      print(f'Read {len(train_examples)} examples in {il}')
      # all_train_examples[il].extend(train_examples)
      # all_train_sro_d[il].update(train_sro_d)

  print('Number of train examples = %d'%len(all_train_examples))
  # complete_train_examples = []
  # for il in il_set:
  #   train_examples = all_train_examples[il]
  #   complete_train_examples.extend(train_examples)
  random.shuffle(all_train_examples)

  train_dataset = SentencesDataset(all_train_examples, model=model)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=FLAGS.batch_size)
  # train_objectives.append((train_dataloader, train_loss))

  queries, corpus, relevant, dev_sroD = process_eval_examples(dev_examples_d)
  print('Number of dev examples = %d' % (len(queries)))
  ir_evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant)

  model.fit(train_objectives=[(train_dataloader, losses.MultipleNegativesRankingLoss(model))],
            epochs=FLAGS.epochs,
            optimizer_params={'lr': FLAGS.learning_rate},
            evaluator=ir_evaluator,
            evaluation_steps=10000,
            output_path=FLAGS.model_save_path,
            save_best_model=True)

  return model

def generate_triple_embs(model, triples):
  print('Embedding Triples...')

  index = faiss.IndexFlatIP(768)
  partition_size = int(3e6)
  num_partitions = int(math.ceil(len(triples)/partition_size))
  for i in range(num_partitions):
    print('Triple Partition: ',i)
    part_triples = triples[i*partition_size:(i+1)*partition_size]
    part_triple_embs = model.encode(
        part_triples,
        batch_size=2048,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True)

    if i == 0:
      index.train(part_triple_embs)
    index.add(part_triple_embs)

  return index

def process_triples(model, triples_d):
  all_lang_triples, all_lang_triple_sro_d, all_lang_indexes = {}, {}, {}

  # if FLAGS.all_langs_test:
  #   triple_path = FLAGS.model_save_path+'/triple_cache_%s_alt.pkl'
  #   index_path = FLAGS.model_save_path+'/index_%s_alt'
  # else:
  triple_path = FLAGS.model_save_path+'/triple_cache_%s.pkl'
  index_path = FLAGS.model_save_path+'/index_%s'

  if FLAGS.build_index:
    all_lang_triples, all_lang_triple_sro_d = language_triples(
        FLAGS.retrieval_languages, triples_d)

    for language in FLAGS.retrieval_languages:
      print('Language: ', language)
      triples, triple_sro_d = all_lang_triples[language], all_lang_triple_sro_d[
          language]
      print('Number of triples = %d' % len(triples))
      print('Generating triple embeddings...')
      index = generate_triple_embs(model, triples)
      all_lang_indexes[language] = index

      if not FLAGS.dont_save_index:
        os.makedirs(FLAGS.model_save_path, exist_ok=True)
        print('Dumping triples to: ', triple_path%language)
        print('Dumping index to: ', index_path%language)
        pickle.dump([triples, triple_sro_d], open(triple_path%language,'wb'))
        faiss.write_index(index, index_path%language)
  else:
    print('Reading from cache...')
    for i, language in enumerate(FLAGS.retrieval_languages):
      print('Reading triples from: ', triple_path%language)
      print('Reading index from: ', index_path%language)
      triples, triple_sro_d = pickle.load(open(triple_path%language, 'rb'))
      index = faiss.read_index(index_path%language)

      if i==0:
        all_lang_triples = triples
        all_lang_triple_sro_d = triple_sro_d
        all_lang_indexes = index
      else:
        all_lang_triples.extend(triples)
        all_lang_triple_sro_d.update(triple_sro_d)
        triple_embs = index.reconstruct_n(0, len(triples))
        all_lang_indexes.add(triple_embs)
      print('Number of triples: ', len(triples))

      # all_lang_triples[language] = triples
      # all_lang_triple_sro_d[language] = triple_sro_d
      # all_lang_indexes[language] = index

  return all_lang_triples, all_lang_triple_sro_d, all_lang_indexes

def read_triples(triples_path):
  # triples_d = pickle.load(open(triples_path,'rb'))
  jlines = [json.loads(line) for line in open(triples_path).readlines()]
  triples_d = {jline['fact']: jline['fact_label'] for jline in jlines}
  proc_triples_d = {}
  for canon_fact, fact_lang_d in triples_d.items():
    if 'ur' in fact_lang_d:
      fact_lang_d['ur'] = ';'.join(fact_lang_d['ur'].split(';')[::-1])
    # if FLAGS.fact_set == 'enil:rel' or FLAGS.fact_set == 'enil:entrel' or FLAGS.fact_set == 'all:entrel' or FLAGS.fact_set == 'canon':
    # if FLAGS.fact_set != 'en' and FLAGS.fact_set != 'en_il' and FLAGS.fact_set != 'all':
      # fact_lang_d = utils.add_languages(fact_lang_d, FLAGS.input_languages, canon_fact)
    # if FLAGS.all_langs_test:
    #   if len({'en','hi','te','ta','gu','ur'}-fact_lang_d.keys()) != 0:
    #     continue
    proc_triples_d[canon_fact] = fact_lang_d

  none_d = {lang: 'None ; None ; None' for lang in set(['en','hi','te','ta','ur','gu','as'])}
  proc_triples_d['Q0;P0;Q0'] = none_d
  return proc_triples_d

def language_triples(languages, triples_d):
  sro_d, triples = {lang: {} for lang in languages
                   }, {lang: [] for lang in languages}
  print('Reading triples file...')
  eng_fact_not_found = 0
  for triple, fact_lang_d in tqdm(triples_d.items()):
    sid, rid, oid = triple.split(';')

    for language in languages:
      if language in fact_lang_d:
        lang_fact = fact_lang_d[language]
        fields = [f.strip() for f in lang_fact.split(';')]
        if len(fields) < 3:
          continue
        triples[language].append(lang_fact)
        sro_d[language][lang_fact] = [
            fields[0], fields[1], [fields[2]], sid, rid, oid, fact_lang_d]

  for language in languages:
    none_fact = 'None ; None ; None'
    triples[language].append(none_fact)
    sro_d[language][none_fact] = ['None', 'None', ['None'], 'Q0', 'P0', 'Q0', {'en': none_fact}]
    triples[language] = list(set(triples[language]))

  print('English Fact not found = ', eng_fact_not_found)
  return triples, sro_d

def write_rank_data(rank_data, output_dir, mode):
  '''Write the Ranker data
  '''
  os.makedirs(output_dir, exist_ok=True)
  if FLAGS.append_rank_data:
    source_path = open(f'{output_dir}/{mode}.source', 'a')
    target_path = open(f'{output_dir}/{mode}.target', 'a')
  else:
    source_path = open(f'{output_dir}/{mode}.source', 'w')
    target_path = open(f'{output_dir}/{mode}.target', 'w')

  for rank_datum in rank_data:
    source_path.write(str(rank_datum[0])+'\n')
    target_path.write(str(rank_datum[1])+'\n')

  source_path.close()
  target_path.close()
  return


# def print_performance(performances, mode):
#   '''Prints the performance and writes it to file for persistence
#   '''
#   ent_str, rel_str, fact_str = '', '', ''
#   subj_str, obj_str = '', ''
#   language_str = ''
#   for i, performance in enumerate(performances):
#     language_str += FLAGS.input_languages[i]+','
#     subj_str += performance[0]+','
#     obj_str += performance[1]+','
#     ent_str += performance[2]+','
#     rel_str += performance[3]+','
#     fact_str += performance[4]+','
#   language_str = language_str.strip()
#   subj_str, obj_str, ent_str, rel_str, fact_str = subj_str.strip(','), obj_str.strip(','), ent_str.strip(','), rel_str.strip(','), fact_str.strip(',')
#   perf_str = f'{mode}\nLanguages,{language_str},\nSubject,{subj_str}\nObject,{obj_str}\nEntity,{ent_str}\nRelation,{rel_str}\nFact,{fact_str}\n'
#   print(perf_str)
#   open(FLAGS.model_save_path+'/Information-Retrieval_evaluation_results.csv','a').write(perf_str+'\n')
#   return


def test(model, 
        test_data, 
        index, 
        triples, 
        triple_sro, 
        canon_desc_d,
        mode):
  '''Run the DE model in test mode
  '''
  all_ranker_data_d = {'fact': [], 'sid': []}
  all_languages, all_retrieved_facts = [], []

  # all_performance, all_languages = [], []
  # all_gold_ids, all_gold_facts, all_gold_en_facts, all_retrieved_facts = [], [], [], []

  # if FLAGS.fact_set == 'all' and 'all' not in triples_d:
  # if 'all' not in triples_d:
  #   language = FLAGS.retrieval_languages[0]
  #   all_lang_triples = copy.copy(triples_d[language])
  #   all_lang_triple_sro_d = copy.copy(triple_sro_d[language])
  #   all_lang_indexes = copy.copy(index_d[language])
  #   del triples_d[language]; del triple_sro_d[language]; del index_d[language]
  #   for li, language in enumerate(FLAGS.retrieval_languages[1:]):
  #     all_lang_triples.extend(triples_d[language])
  #     all_lang_triple_sro_d.update(triple_sro_d[language])
  #     triple_embs = index_d[language].reconstruct_n(0, len(triples_d[language]))
  #     all_lang_indexes.add(triple_embs)
  #     del triples_d[language]; del triple_sro_d[language]; del index_d[language]
  #   triples_d['all'] = all_lang_triples
  #   triple_sro_d['all'] = all_lang_triple_sro_d
  #   index_d['all'] = all_lang_indexes

  # if mode == 'test' and not FLAGS.debug:
  #   languages = indiclink_languages
  # else:
  #   languages = FLAGS.input_languages

  for language in FLAGS.input_languages:
    # test_data_lang = test_data[language]
    # retrieval_languages = utils.retrieval_languages(FLAGS.fact_set, language)
    # assuming it will be only 1 fact language
    # triple_embs, triples, triple_sro_dict = index_d[retrieval_languages[0]], triples_d[retrieval_languages[0]], triple_sro_d[retrieval_languages[0]]
    # performance, ranker_data_d, gold_facts, gold_en_facts, retrieved_facts = retrieve_facts(
    ranker_data_d, retrieved_facts = retrieve_facts(test_data[language], model, index, triple_sro,
        canon_desc_d, FLAGS.topk, triples, language, mode)

    for key in all_ranker_data_d:
      all_ranker_data_d[key].extend(ranker_data_d[key])

    all_languages.extend([language]*len(retrieved_facts))
    # all_performance.append(performance)
    # all_gold_facts.extend(gold_facts)
    # all_gold_en_facts.extend(gold_en_facts)
    all_retrieved_facts.extend(retrieved_facts)

  # return all_performance, all_ranker_data_d, all_languages, all_gold_facts, all_gold_en_facts, all_retrieved_facts
  return all_ranker_data_d, all_languages, all_retrieved_facts


def ranker_data(rank_dir, modes, model, data_d, index, triples, triple_sro_d, canon_desc_d):
  '''Calls helper function ranker_data_ with different modes
  '''
  for mode in modes.split('.'):
    print('Mode: ', mode)
    ranker_data_d, all_languages, retrieved_facts = test(model, data_d[mode], index, triples, triple_sro_d, canon_desc_d, mode)
    write_rank_data(ranker_data_d['fact'], f'{rank_dir}/fact/', mode)
    write_rank_data(ranker_data_d['sid'], f'{rank_dir}/sid', mode)
    open(f'{rank_dir}/{mode}_retrieved_facts.txt', 'w').write('\n'.join(retrieved_facts)+'\n')
    open(f'{rank_dir}/{mode}_languages.txt','w').write('\n'.join(all_languages)+'\n')
    # ranker_data_(rank_dir, mode, model, data_d[mode], index, triples, triple_sro_d, canon_desc_d)

# def ranker_data_(rank_dir, mode, model, data, index, triples, triple_sro_d, canon_desc_d):
#   '''1. Gets results of DE and generate input data for ranker model
#      2. Writes the ranker input data
#   '''
#   # performances, ranker_data_d, all_languages, gold_facts, gold_en_facts, retrieved_facts = test(
#   ranker_data_d, all_languages, retrieved_facts = test(model, data, index, triples, triple_sro_d, canon_desc_d, mode)

#   # if not FLAGS.dont_save_rank_data:
#   write_rank_data(ranker_data_d['fact'], f'{rank_dir}/fact/', mode)
#   write_rank_data(ranker_data_d['sid'], f'{rank_dir}/sid', mode)
#   open(f'{rank_dir}/{mode}_retrieved_facts.txt', 'w').write('\n'.join(retrieved_facts))

#     # if FLAGS.append_rank_data:
#     #   open(f'{rank_dir}/{mode}_languages.txt','a').write('\n'.join(all_languages)+'\n')
#     #   open(f'{rank_dir}/{mode}_gold_facts.txt', 'a').write('\n'.join(gold_facts)+'\n')
#     #   open(f'{rank_dir}/{mode}_gold_en_facts.txt', 'a').write('\n'.join(gold_en_facts)+'\n')
#     #   open(f'{rank_dir}/{mode}_retrieved_facts.txt', 'a').write('\n'.join(retrieved_facts)+'\n')
#     # else:
#       # open(f'{rank_dir}/{mode}_languages.txt','w').write('\n'.join(all_languages))
#       # open(f'{rank_dir}/{mode}_gold_facts.txt', 'w').write('\n'.join(gold_facts))
#       # open(f'{rank_dir}/{mode}_gold_en_facts.txt', 'w').write('\n'.join(gold_en_facts))
#       # open(f'{rank_dir}/{mode}_retrieved_facts.txt', 'w').write('\n'.join(retrieved_facts))

#   # print_performance(performances, mode)
#   return


def main(_):
  # FLAGS.de_mode = FLAGS.de_mode.split('.')
  # FLAGS.input_languages = ['en']

  # FLAGS.input_languages = utils.split_languages(FLAGS.input_languages)
  # FLAGS.retrieval_languages = []
  # for language in FLAGS.input_languages:
  #   FLAGS.retrieval_languages.extend(utils.retrieval_languages(FLAGS.fact_set, language))
  # FLAGS.retrieval_languages = list(set(FLAGS.retrieval_languages))
  # if 'all' in FLAGS.retrieval_languages:
  #   FLAGS.retrieval_languages = FLAGS.input_languages

  ranker_save_path = f'{FLAGS.model_save_path}/s2s_k{FLAGS.topk}_{FLAGS.ranker_languages}'
  os.makedirs(ranker_save_path, exist_ok=True)

  # if FLAGS.model_load_path:
  #   FLAGS.model_load_path = os.path.join(FLAGS.model_dir, FLAGS.model_load_path)
  # FLAGS.model_save_path = os.path.join(FLAGS.model_dir, FLAGS.model_save_path)

  # input_languages, retrieval_languages = [], []

  # for i_language in FLAGS.input_languages:
  #   fl = utils.retrieval_languages(FLAGS.fact_set, i_language)
  #   if fl[0] == 'all':
  #     fl = FLAGS.input_languages
  #   if i_language == 'as': # No training data for assamese
  #     continue
  #   for j_language in fl:
  #     input_languages.append(i_language)
  #     retrieval_languages.append(j_language)
  
  # FLAGS.input_languages = indiclink_languages
  FLAGS.input_languages = ['en']
  FLAGS.retrieval_languages = indiclink_languages
  if FLAGS.debug:
    # FLAGS.model_save_path = FLAGS.model_save_path+'/dummy'
    FLAGS.retrieval_languages = ['ur']

  if not FLAGS.triples_path:
    triples_path = f'{FLAGS.data_dir}/WikiData_facts.jsonl'
  else:
    triples_path = FLAGS.triples_path
  if not FLAGS.train_path:
    train_path = f'{FLAGS.data_dir}/train.jsonl'
  else:
    train_path = FLAGS.train_path
  if not FLAGS.test_path:
    test_path = f'{FLAGS.data_dir}/IndicLink_release.jsonl'
  else:
    test_path = FLAGS.test_path

  canon_desc_d = read_triples(triples_path)

  train_examples, dev_examples = read_jsonl(train_path, canon_desc_d, test=False)
  test_examples = read_jsonl(test_path, canon_desc_d, test=True)

  if 'train' in FLAGS.de_mode:
    model = load_model()
    train(model, train_examples, dev_examples)

  if 'test' in FLAGS.de_mode:
    # Reload the best-model after training
    FLAGS.model_load_path = FLAGS.model_save_path
    model = load_model()

    triples, triple_sro_d, index = process_triples(model, canon_desc_d)

    ranker_data(ranker_save_path, FLAGS.ranker_mode, model, 
                {'test': test_examples, 'dev': dev_examples, 'train': train_examples},
                index, triples, triple_sro_d, canon_desc_d)

    print('Ranker Data Saved Path: ', ranker_save_path)


if __name__ == '__main__':
  app.run(main)