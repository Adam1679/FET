import socket
from os.path import join
from platform import platform

if platform().startswith('Windows'):
    PLATFORM = 'Windows'
    DATA_DIR = 'd:/data/fetel-data'
elif platform().startswith('Linux-3.10.0-957.el7.x86_64-x86_64-with-glibc2.10'):
    PLATFORM = 'Tian He'
    DATA_DIR = '/GPUFS/swufe_fxiao_5/adam/FET'
elif platform().startswith('Linux'):
    PLATFORM = 'Linux'
    DATA_DIR = '/home/ubuntu/IFETEL'
else:
    PLATFORM = 'MACOX'
    DATA_DIR = '/Users/adam/Desktop/FET/IFETEL'

print(PLATFORM)
TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

RANDOM_SEED = 771
NP_RANDOM_SEED = 7711
PY_RANDOM_SEED = 9973

MACHINE_NAME = socket.gethostname()
RES_DIR = join(DATA_DIR, 'res')
EL_DATA_DIR = join(DATA_DIR, 'el')
MODEL_DIR = join(DATA_DIR, 'models')
LOG_DIR = join(DATA_DIR, 'log')

EL_CANDIDATES_DATA_FILE = join(RES_DIR, 'enwiki-20151002-candidate-gen.pkl')
WIKI_FETEL_WORDVEC_FILE = join(RES_DIR, 'enwiki-20151002-nef-wv-glv840B300d.pkl')
WIKI_ANCHOR_SENTS_FILE = join(RES_DIR, 'enwiki-20151002-anchor-sents.txt')

FIGER_FILES = {
    'typed-wiki-mentions': join(DATA_DIR, 'Wiki/enwiki-20151002-anchor-mentions-typed.txt'),  # no data
    'anchor-train-data-prefix': join(DATA_DIR, 'Wiki/enwiki20151002anchor-fetwiki-0_1'),  # no data
    'type-vocab': join(DATA_DIR, 'Wiki/figer-type-vocab.txt'),  # yes
    'wid-type-file': join(DATA_DIR, 'Wiki/wid-types-figer.txt'),  # yes
    'fetel-test-mentions': join(DATA_DIR, 'Wiki/figer-fetel-test-mentions.json'),  # yes
    'fetel-test-sents': join(DATA_DIR, 'Wiki/figer-fetel-test-sents.json'),  # yes
    'noel-typing-results' : join (DATA_DIR, 'Wiki/noel-fet-results-aaa-figer.txt'),  # yes,
    'type-emb' : join (DATA_DIR, 'result/type_embedding_figer')
}

BBN_FILES = {
    'typed-wiki-mentions': join(DATA_DIR, 'BBN/enwiki-20151002-anchor-mentions-typed.txt'),
    'anchor-train-data-prefix': join(DATA_DIR, 'BBN/enwiki20151002anchor-fetbbn-0_1'),
    'type-vocab': join(DATA_DIR, 'BBN/bbn-type-vocab.txt'),
    'wid-type-file': join(DATA_DIR, 'BBN/wid-types-bbn.txt'),
    'fetel-test-mentions': join(DATA_DIR, 'BBN/bbn-fetel-test-mentions.json'),
    'fetel-test-sents': join(DATA_DIR, 'BBN/bbn-fetel-test-sents.json'),
    'noel-typing-results' : join (DATA_DIR, 'BBN/noel-fet-results-aaa-bbn.txt'),
    'type-emb' : join (DATA_DIR, 'result/type_embedding_bnn')
}
