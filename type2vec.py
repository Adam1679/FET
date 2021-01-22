import datetime
import logging
import os

import numpy as np
import torch

import config
from modelexp import fetelexp, exputils
from utils.loggingutils import init_universal_logging


def train_model() :
    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    data_prefix = datafiles['anchor-train-data-prefix']
    train_data_pkl = data_prefix + '-train.pkl'
    dev_data_pkl = data_prefix + '-dev.pkl'
    embedding_path = "./result/type_embedding_{}".format (dataset)
    type2vec_model_path = "./result/type_embedding_model_{}".format (dataset)
    type2vec_graoh_path = "./result/type_embedding_graph_{}".format (dataset)
    el_candidates_file = config.EL_CANDIDATES_DATA_FILE
    print ('init el with {} ...'.format (el_candidates_file), end=' ', flush=True)
    print ('done', flush=True)

    # 存储了所有的token的embedding， type的名字，id， 还有一些parent type的名字，id
    gres = exputils.GlobalRes (datafiles['type-vocab'], word_vecs_file, datafiles['type-emb'])

    logging.info ('dataset={} {}'.format (dataset, data_prefix))
    fetelexp.get_type_vec (gres, dev_data_pkl, embedding_path, graph_save_path=type2vec_graoh_path,
                           model_save_path=type2vec_model_path)


if __name__ == '__main__' :
    import random

    torch.random.manual_seed (config.RANDOM_SEED)
    np.random.seed (config.NP_RANDOM_SEED)
    random.seed (config.PY_RANDOM_SEED)
    str_today = datetime.date.today ().strftime ('%y-%m-%d')
    log_file = os.path.join (config.LOG_DIR, '{}-{}-{}.log'.format (os.path.splitext (
        os.path.basename (__file__))[0], str_today, config.MACHINE_NAME))
    init_universal_logging (log_file, mode='a', to_stdout=True)

    device = torch.device ('cuda') if torch.cuda.device_count () > 0 else torch.device ('cpu')
    train_model ()
