import json
import ipdb
import os
import re
from absl import flags
from absl import app
import pickle
import numpy as np

from collections import Counter

'''Compute scores of all the different models
'''

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', 'Fact Model path')
flags.DEFINE_string('predicate2wikidataID', '../data/predicate_id.pkl', 'predicate mapping dict')
flags.DEFINE_string('entity2wikidataID', '../data/entity_id.pkl', 'entity mapping dict')
flags.DEFINE_integer('topk', 1, 'Top-k retrieved facts')

flags.DEFINE_boolean('free_generation', False, 'Dont use constrained generation')
flags.DEFINE_boolean('retrieval', False, 'Evaluate performance of retrieval')
flags.DEFINE_boolean('recall', False, 'Computes recall')
flags.DEFINE_boolean('gold_macro', False, 'Macro precision w.r.t gold relations')
flags.DEFINE_boolean('write_results', False, 'Write GT and predictions for analysis')

def process_jsonl(fp):
    json_f = open(fp, 'r')
    json_dict = [json.loads(l) for l in json_f.readlines()]
    json_dict = sorted(json_dict, key=lambda x: x['id'])
    return json_dict


def main(_):

  languages_f = open(f'{FLAGS.model_dir}/../test_languages.txt','r')
  languages = [line.strip('\n') for line in languages_f]
  
  print('Loding predicate_id_d and entity_id_d...')
  predicate_id_d = pickle.load(open(FLAGS.predicate2wikidataID, 'rb'))
  entity_id_d = pickle.load(open(FLAGS.entity2wikidataID, 'rb'))
  entity_id_d[''] = 'Q0'
  predicate_id_d[''] = 'P0'
  entity_id_d['None'] = 'Q0'
  predicate_id_d['None'] = 'P0'

  gt_f = open(f'{FLAGS.model_dir}/../test_gold_facts.txt','r')    
  gt_lines = [[l.strip() for l in line.strip('\n').split('[SEP]')] for line in gt_f]

  ret_f = open(f'{FLAGS.model_dir}/../test_retrieved_facts.txt','r')    
  ret_lines = [[l.strip() for l in line.strip('\n').split('[SEP]')] for line in ret_f]

  if FLAGS.write_results:
    print(f'Writing results to: {FLAGS.model_dir}/results.txt')
    results_f = open(f'{FLAGS.model_dir}/results.txt','w')

  if FLAGS.free_generation:
    facts_fp = f'{FLAGS.model_dir}/free-fact-predictions.jsonl'
  else:
    facts_fp = f'{FLAGS.model_dir}/const-fact-predictions.jsonl'

  if FLAGS.model_dir and os.path.exists(facts_fp):       
    print(f'### Reading {facts_fp}')
    fact_list = process_jsonl(facts_fp)
  else:
    fact_list = [None]*len(languages)
  
  total = 0.
  lang_sum = {}
  lang_total = {}

  print('Done loading!')
  incorrect_format, incorrect_entities, incorrect_facts = 0, 0, 0
  total_facts, no_facts = 0, 0

  pred_rel_counter, gt_rel_counter = [], []
  all_pred_es, all_pred_rs = [], []

  macro_d = {}
  if FLAGS.gold_macro:
    for gt_raw in gt_lines:
      for gt_fact in gt_raw:
        gt_pred_id = gt_fact.split(';')[1]
        if gt_pred_id not in macro_d:
          macro_d[gt_pred_id] = [0, 0]
        macro_d[gt_pred_id][1] += 1

  total_gt_facts = 0
  none_preds = 0

  for fact, lang, gt_raw, ret_raw in zip(fact_list, languages, gt_lines, ret_lines):

    gt_subjs, gt_preds, gt_objs, gt_facts = set(), set(), set(), set()

    for gt_fact in gt_raw:
      gt_subj_id, gt_pred_id, gt_obj_id = gt_fact.split(';')
      if gt_pred_id == 'P0':
        gt_subj_id, gt_obj_id = 'Q0', 'Q0'
      gt_rel_counter.append(gt_pred_id)
      gt_subjs.add(gt_subj_id); gt_preds.add(gt_pred_id); gt_objs.add(gt_obj_id)
      gt_facts.add(f'{gt_subj_id};{gt_pred_id};{gt_obj_id}')

    total_gt_facts += len(gt_facts)

    ret_subjs, ret_preds, ret_objs, ret_facts = [], [], [], []
    for ret_fact in ret_raw:
      ret_fields = ret_fact.split(';')
      if len(ret_fields) != 3:
        continue
      ret_subj_id, ret_pred_id, ret_obj_id = ret_fields 
      ret_subjs.append(ret_subj_id); ret_preds.append(ret_pred_id); ret_objs.append(ret_obj_id)
      ret_facts.append(f'{ret_subj_id};{ret_pred_id};{ret_obj_id}')

    if lang not in lang_sum:
      lang_sum[lang] = [0, 0, 0, 0, 0]      
    if lang not in lang_total:
      lang_total[lang] = np.array([0, 0, 0, 0, 0])
    
    if FLAGS.recall:
      lang_total[lang] += [len(gt_subjs), len(gt_objs), len(gt_preds), len(gt_subjs.union(gt_objs)), len(gt_facts)]
      total += len(gt_facts)
    else:
      lang_total[lang] += 1
      total += 1

    subj_bool, obj_bool, rel_bool = False, False, False
    if fact != None:
      pred_facts = []

      if FLAGS.write_results:
        results_f.write('\n\n')
        # results_f.write('Sentence id: '+str(sid)+'\n')
        results_f.write('Sentence: '+fact['input']+'\n')
        # results_f.write('Gold: \n')
        # for gt_en_fact, gt_canon_fact in zip(gt_en, gt_facts):
        #   results_f.write(gt_en_fact+'\t'+gt_canon_fact+'\n')
        results_f.write('Predictions: \n')

      for pred_i, prediction in enumerate(fact['predictions']):
        pf = prediction['text'].split(';')

        total_facts += 1
        if len(pf) != 3:
          if pred_i == 0:
            incorrect_format += 1
          pf = ['Q-1','P-1','Q-1']
        subj_title, pred_title, obj_title = ';'.join(pf[:-2]), pf[-2], pf[-1]
        obj_title = obj_title.split('>>')[0]
        subj_title, pred_title, obj_title = subj_title.strip(), pred_title.strip(), obj_title.strip()

        if pred_i == 0:
          pred_rel_counter.append(pred_title)
          all_pred_es.append(subj_title)
          all_pred_es.append(obj_title)
          all_pred_rs.append(pred_title)

        if pred_title in predicate_id_d:
          pred_relation = predicate_id_d[pred_title]
        else:
          pred_relation = 'P-1'

        if subj_title in entity_id_d:
          pred_subject = entity_id_d[subj_title]
        else:
          pred_subject = 'Q-1'

        if obj_title in entity_id_d:
          pred_object = entity_id_d[obj_title]
        else:
          pred_object = 'Q-1'

        pred_fact = f'{pred_subject};{pred_relation};{pred_object}'
        if '-1' in pred_fact and pred_i == 0:
          incorrect_entities += 1

        # if pred_i == 0 and pred_fact != 'Q0;P0;Q0' and pred_fact not in facts:
        #   incorrect_facts += 1

        pred_facts.append(pred_fact)
        if FLAGS.write_results:
          results_f.write(prediction['text']+'\t'+pred_fact+'\n')


    if FLAGS.retrieval:
      pred_facts = ret_facts

    seen_pred_subjs, seen_pred_rels, seen_pred_objs = set(), set(), set()
    dedup_pred_facts = []
    for p in pred_facts:
      if p not in dedup_pred_facts:
        dedup_pred_facts.append(p)
    pred_facts = dedup_pred_facts

    if FLAGS.gold_macro:
      for gold_fact in gt_facts:
        _, gold_rel, _ = gold_fact.split(';')
        for pred_fact in pred_facts:
          if pred_fact in set(gt_facts)-set([gold_fact]):
            continue
          break
        if pred_fact == gold_fact:
          macro_d[gold_rel][0] += 1
    
    if pred_facts[0] == 'Q0;P0;Q0':
      none_preds += 1

    for pred_fact in set(pred_facts[:FLAGS.topk]):
      pred_subject, pred_relation, pred_object = pred_fact.split(';')
      if pred_subject in gt_subjs and pred_subject not in seen_pred_subjs:
        subj_bool = True
        lang_sum[lang][0] += 1
        seen_pred_subjs.add(pred_subject)
      else:
        subj_bool = False

      if pred_object in gt_objs and pred_object not in seen_pred_objs:
        obj_bool = True
        lang_sum[lang][1] += 1
        seen_pred_objs.add(pred_object)
      else:
        obj_bool = False

      if pred_relation in gt_preds and pred_relation not in seen_pred_rels:
        lang_sum[lang][2] += 1
        seen_pred_rels.add(pred_relation)
              
      if subj_bool and obj_bool:
        lang_sum[lang][3] += 1
            
      if pred_fact in gt_facts:
        if FLAGS.write_results:
          results_f.write('Correct\n')
        lang_sum[lang][4] += 1
      else:
        if FLAGS.write_results:
          results_f.write('Wrong\n')

  print('Total number of None predictions: ', none_preds)  
  
  sum_subj, sum_obj, sum_rel, sum_fact, sum_ent, total = 0, 0, 0, 0, 0, 0
  performances, languages = [], []
  for lang in lang_sum:
    subj_correct, obj_correct, rel_correct, ent_correct, fact_correct = lang_sum[lang]
    sum_ent += ent_correct
    sum_subj += subj_correct
    sum_obj += obj_correct
    sum_rel += rel_correct
    sum_fact += fact_correct
    total_subj, total_obj, total_pred, total_ent, total_lang = lang_total[lang]
    total += total_lang
    subj_acc = subj_correct/total_subj*100
    obj_acc = obj_correct/total_obj*100
    rel_acc = rel_correct/total_pred*100
    ent_acc = ent_correct/total_ent*100
    fact_acc = fact_correct/total_lang*100
    performances.append(['%.1f'%subj_acc, '%.1f'%obj_acc, 
                          '%.1f'%rel_acc, '%.1f'%ent_acc, '%.1f'%fact_acc])
    languages.append(lang)

  lang_str, subj_str, obj_str, ent_str, rel_str, fact_str = 'Languages,','Subject,','Object,','Entity,','Relation,','Fact,'
  for language, performance in zip(languages, performances):
    lang_str += language+','
    subj_str += performance[0]+','
    obj_str += performance[1]+','
    rel_str += performance[2]+','
    ent_str += performance[3]+','
    fact_str += performance[4]+','
  lang_str = lang_str+'Avg.'
  subj_str = subj_str+'%.1f'%(100*sum_subj/total)
  obj_str = obj_str+'%.1f'%(100*sum_obj/total)
  rel_str = rel_str+'%.1f'%(100*sum_rel/total)
  ent_str = ent_str+'%.1f'%(100*sum_ent/total)
  fact_str = fact_str+'%.1f'%(100*sum_fact/total)
  perf_str = f'{lang_str}\n{subj_str}\n{obj_str}\n{ent_str}\n{rel_str}\n{fact_str}'
  
  avg_macro = 0
  for pred in macro_d:
    avg_macro += macro_d[pred][0] / macro_d[pred][1] 
  if FLAGS.gold_macro:
    avg_macro = avg_macro / len(macro_d)
    print('Average Macro Precision = %.2f'%(avg_macro*100))

  if FLAGS.write_results:
    results_f.close()

  print(perf_str)

  print('Total Examples = ', total)
  print(f'Total Facts: {total_facts}, Incorrect Facts: {incorrect_facts}, Incorrect Entities: {incorrect_entities}, Incorrect Format: {incorrect_format}')
  print('Total number of GT facts = ', total_gt_facts)

if __name__ == '__main__':
    app.run(main)
