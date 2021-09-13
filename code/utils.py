import os
import copy
import ipdb
import pickle
from tqdm import tqdm

def split_languages(languages_str):
  if '_' in languages_str:
    lset, suffix = languages_str.split('_')
  else:
    lset, suffix = languages_str, ''

  if lset == 'indiclink':
    lset = ['en', 'hi', 'te', 'ta', 'ur', 'gu', 'as']
  else:
    lset = languages_str.split('.')
  
  if suffix:
    for i in range(len(lset)):
      lset[i] = lset[i]+'_'+suffix
   
  return lset

def retrieval_languages(fact_set, input_language):
  fact_languages = set([])
  for part in fact_set.split('_'):
    if part == 'en':
      fact_languages.add('en')
    if part == 'il':
      fact_languages.add(input_language)
    if part == 'enil:rel':
      fact_languages.add(input_language+'_enil:rel')
    if part == 'enil:entrel':
      fact_languages.add(input_language+'_enil:entrel')
    if part == 'all:entrel':
      fact_languages.add(input_language+'_all:entrel')
    if part == 'all':
      fact_languages.add('all')

  return list(fact_languages)
    

def add_languages(fact_lang_d, input_languages, canon_fact):
  pro_fact_lang_d = copy.copy(fact_lang_d)
  en_fields = None
  en_subj, en_rel, en_obj = '', '', ''
  all_langs_subj, all_langs_pred, all_langs_obj = '', '', ''
  if 'en' in fact_lang_d:
    en_fields = [f.strip() for f in fact_lang_d['en'].split(';')]
    en_subj = ''.join(en_fields[:-2])
    en_rel = en_fields[-2]
    en_obj = en_fields[-1]
    all_langs_subj = f'{en_subj}'
    all_langs_pred = f'{en_rel}'
    all_langs_obj = f'{en_obj}'
  
  for language in input_languages:    
    lsubj, lrel, lobj, lfields = '', '', '', None
    lang_fact = ''
    if language in fact_lang_d:
      lang_fact = fact_lang_d[language]
      lfields = [f.strip() for f in lang_fact.split(';')]
      lsubj = ''.join(lfields[:-2])
      lrel = lfields[-2]
      lobj = lfields[-1]      

      all_langs_subj += f' : {lsubj}'
      all_langs_pred += f' : {lrel}'
      all_langs_obj += f' : {lobj}'

    if en_fields or lfields:
      pro_fact_lang_d[language+'_enil:rel'] = f'{en_subj}; {en_rel} : {lrel}; {en_obj}'
      pro_fact_lang_d[language+'_enil:entrel'] = f'{en_subj} : {lsubj}; {en_rel} : {lrel}; {en_obj} : {lobj}'      

  pro_fact_lang_d['canon'] = canon_fact
  for language in input_languages:
    pro_fact_lang_d[language+'_all:entrel'] = f'{all_langs_subj}; {all_langs_pred}; {all_langs_obj}'

  return pro_fact_lang_d

