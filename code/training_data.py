import json
import ipdb
import regex as re
from tfrecord_lite import tf_record_iterator
from collections import Counter

### Create training data in English from WebRED

it = tf_record_iterator('../data/WebRED/webred_21.tfrecord')
outf = open('../data/train.jsonl','w')

examples = {}
all_relations = []
for record in it:
    sentence = record['sentence'][0].decode('utf-8')
    sentence = re.sub('SUBJ\{(.*?)\}', r'\1', sentence)
    sentence = re.sub('OBJ\{(.*?)\}', r'\1', sentence)
    source = record['source_name'][0].decode('utf-8') 
    relation = record['relation_name'][0].decode('utf-8')
    target = record['target_name'][0].decode('utf-8')

    if (record['num_pos_raters']/record['num_raters'])[0] < 0.5:
        relation = 'None'
    all_relations.append(relation)

    fact = source+' ; '+relation+' ; '+target

    if sentence in examples:
        examples[sentence].append(fact)
    else:
        examples[sentence] = [fact]

total_facts = 0
for i, sentence in enumerate(examples):
    facts = examples[sentence]
    jline = {'language': 'en', 'sent_id':i, 'sentence':sentence, 'facts':[{'fact': fact} for fact in facts]}
    outf.write(json.dumps(jline)+'\n')
    total_facts += len(facts)

outf.close()
print('Written training data to: ', outf.name)

print('Total number of sentences = %d'%len(examples))
print('Number of facts per sentence = %f'%(total_facts/len(examples)))

# Statistics of relations
# rel_counter = Counter(all_relations)
# print(rel_counter)
# print(len(rel_counter))