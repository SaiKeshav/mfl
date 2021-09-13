from genre.fairseq_model import mGENRE
from fairseq.models.bart import BARTModel
from genre.trie import Trie, MarisaTrie

from absl import flags
from absl import app

from tqdm import tqdm
import ipdb
import json
import pickle

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', 'Seq2Seq model directory')
flags.DEFINE_string('input_path', '', 'Input to Seq2Seq model')
flags.DEFINE_string('output_path', '', 'Output of Seq2Seq model')
flags.DEFINE_integer('beams', 5, 'Beam Search')
flags.DEFINE_string('checkpoint_file', 'checkpoint_best.pt', 'Checkpoint to use')

def main(_):
  # model = BARTModel.from_pretrained(FLAGS.model_dir, 
  #                         FLAGS.checkpoint_file, 
  #                         bpe='sentencepiece', 
  #                         sentencepiece_model=FLAGS.model_dir+'/spm_256000.model').eval().to('cuda')

  # outf = open(FLAGS.output_path,'w')  
  # lines = open(FLAGS.input_path,'r').readlines()
  # all_results = model.sample(sentences=lines, beam=FLAGS.beams)
  # for line, result in zip(lines, all_results):
  #   jsonl = {'input': line, 'predictions': [result]}
  #   outf.write(json.dumps(jsonl)+'\n')
  # outf.close()    

  # model = mGENRE.from_pretrained('../models/fairseq_multilingual_entity_disambiguation', 'model.pt').eval().to('cuda')
  trie = pickle.load(open('../data/fact_trie.pkl','rb'))

  # model = mGENRE.from_pretrained('../models/mbart.cc25.v2', sentencepiece_model="sentence.bpe.model", checkpoint_file='model.pt').eval().to('cuda')
  model = mGENRE.from_pretrained('../models/mbart.cc25.v2', sentencepiece_model="sentence.bpe.model", checkpoint_file='model.pt')

  # trie = MarisaTrie([[2]+model.encode('None ; None ; None >> en').tolist()[1:]])
  results = model.sample(sentences=['hello world'], beam=1,
                        prefix_allowed_tokens_fn=lambda batch_id, sent: [
                        e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
                        ])

  # results = model.sample(sentences=['hello world'])
  ipdb.set_trace()

        
if __name__ == '__main__':
  app.run(main)

