import logging
import time
from typing import List

import numpy as np
import torch

from modelexp import exputils
from modelexp.exputils import ModelSample, anchor_samples_to_model_samples, model_samples_from_json
from models.feteldeep import NoName
from models.fetentvecutils import ELDirectEntityVec, MentionFeat
from utils import datautils, utils


def __get_l2_org_type_ids(type_vocab) :
    org_type_ids = list ()
    for i, t in enumerate (type_vocab) :
        if t.startswith ('/organization') and t != '/organization' :
            org_type_ids.append (i)
    return org_type_ids

def __get_l2_person_type_ids(type_vocab):
    person_type_ids = list()
    for i, t in enumerate(type_vocab):
        if t.startswith('/person') and t != '/person':
            person_type_ids.append(i)
    return person_type_ids


def __get_entity_vecs_for_samples(el_entityvec: ELDirectEntityVec, samples: List[ModelSample], noel_pred_results,
                                  filter_by_pop=False, person_type_id=None, person_l2_type_ids=None, type_vocab=None):
    mstrs = [s.mention_str for s in samples]
    prev_pred_labels = None
    if noel_pred_results is not None:
        prev_pred_labels = [noel_pred_results[s.mention_id] for s in samples]
    return el_entityvec.get_entity_vecs(
        mstrs, prev_pred_labels, filter_by_pop=filter_by_pop, person_type_id=person_type_id,
        person_l2_type_ids=person_l2_type_ids, type_vocab=type_vocab)


def __get_entity_vecs_for_mentions(el_entityvec: ELDirectEntityVec, mentions, noel_pred_results, n_types,
                                   filter_by_pop=False):
    all_entity_type_vecs = -np.ones((len(mentions), n_types), np.float32)
    all_el_sgns = np.zeros(len(mentions), np.float32)
    all_probs = np.zeros(len(mentions), np.float32)
    mention_id_to_idxs = {m['mention_id']: i for i, m in enumerate(mentions)}
    doc_mentions_dict = utils.json_objs_to_kvlistdict(mentions, 'file_id')
    for doc_id, doc_mentions in doc_mentions_dict.items():
        prev_pred_labels = [noel_pred_results[m['mention_id']] for m in doc_mentions]
        mstrs = [m['str'] for m in doc_mentions]
        entity_type_vecs, el_sgns, probs = el_entityvec.get_entity_vecs(mstrs, prev_pred_labels,
                                                                        filter_by_pop=filter_by_pop)
        # print(entity_type_vecs.shape)
        for m, vec, el_sgn, prob_vec in zip(doc_mentions, entity_type_vecs, el_sgns, probs):
            idx = mention_id_to_idxs[m['mention_id']]
            # print(vec.shape)
            all_entity_type_vecs[idx] = vec
            all_el_sgns[idx] = el_sgn
            all_probs[idx] = prob_vec
    return all_entity_type_vecs, all_el_sgns, all_probs


def __print_type_freq(labels) :
    import pandas as pd
    pd.set_option ('display.max_rows', 500)
    lis = []
    for v in labels.values () :
        lis.extend (v)
    print (pd.Series (lis).value_counts (normalize=True))

def eval_data(device, gres: exputils.GlobalRes, el_entityvec: ELDirectEntityVec, train_samples_pkl,
                dev_samples_pkl, test_mentions_file, test_sents_file, test_noel_preds_file, type_embed_dim,
                context_lstm_hidden_dim, learning_rate, batch_size, n_iter, dropout, rand_per, per_penalty,
                use_mlp=False, pred_mlp_hdim=None, save_model_file=None, nil_rate=0.5,
                single_type_path=False, stack_lstm=False, concat_lstm=False, results_file=None) :

    # 每个sample都是一个长度为7的tuple：

    dev_samples = datautils.load_pickle_data (dev_samples_pkl)
    # batch_samples：针对每一个mention，context的token sequence id，包括parent的full types
    dev_samples = anchor_samples_to_model_samples (dev_samples, gres.mention_token_id,
                                                   gres.parent_type_ids_dict)  # type: List[LabeledModelSample]
    start = time.time ()
    print ("train_samples_with_label takes {}s".format ((time.time () - start) / 1000))
    lr_gamma = 0.8
    eval_batch_size = 32
    dev_true_labels_dict = {s.mention_id : [gres.type_vocab[l] for l in s.labels] for s in dev_samples}
    # number of class in dev

    # dev_entity_vecs： (2000, 128)
    dev_entity_vecs, dev_el_sgns, dev_el_probs = __get_entity_vecs_for_samples (el_entityvec, dev_samples, None)
    test_samples = model_samples_from_json (gres.token_id_dict,
                                            gres.unknown_token_id,
                                            gres.mention_token_id,
                                            gres.type_id_dict,
                                            test_mentions_file,
                                            test_sents_file)

    test_noel_pred_results = datautils.read_pred_results_file (test_noel_preds_file)

    test_mentions = datautils.read_json_objs (test_mentions_file)
    test_entity_vecs, test_el_sgns, test_el_probs = __get_entity_vecs_for_mentions (el_entityvec,
                                                                                    test_mentions,
                                                                                    test_noel_pred_results,
                                                                                    gres.n_types)

    test_true_labels_dict = {m['mention_id'] : m['labels'] for m in test_mentions} if (
            'labels' in next (iter (test_mentions))) else None

    print ("freq in test")
    __print_type_freq (test_true_labels_dict)
    print ("freq in dev")
    __print_type_freq (dev_true_labels_dict)
    # train_samples = datautils.load_pickle_data (train_samples_pkl)
    # train_samples_with_label = anchor_samples_to_model_samples (train_samples, gres.mention_token_id,
    #                                                             gres.parent_type_ids_dict)  # type: List[LabeledModelSample]
    #
    # del train_samples
    # tr_c = np.array([len (s.labels) for s in train_samples_with_label])
    dev_c = np.array([len(s.labels) for s in dev_samples])

    test_c = np.array ([len (m['labels']) for m in test_mentions])
    # print("avg # in tr/dev/te: {:.4f}/{:.4f}/{:.4f}".format (tr_c.mean(), dev_c.mean(), test_c.mean()))
    # print("min # in tr/dev/te: {:.4f}/{:.4f}/{:.4f}".format (tr_c.min(), dev_c.min(), test_c.min()))
    # print("max # in tr/dev/te: {:.4f}/{:.4f}/{:.4f}".format (tr_c.max(), dev_c.max(), test_c.max()))
    #
    te_entity_vecs, _, te_el_probs = __get_entity_vecs_for_samples (el_entityvec, test_samples, None, True)
    te_el_probs_dict = {test_samples[i].mention_id : te_el_probs[i] for i in range (len (test_samples))}
    dv_entity_vecs, _, dv_el_probs = __get_entity_vecs_for_samples (el_entityvec, dev_samples, None, True)
    dv_el_probs_dict = {dev_samples[i].mention_id : dv_el_probs[i] for i in range (len (dev_samples))}
    pred_labels_dict_dv = {}
    pred_labels_dict_te = {}
    cnt = 0
    for j, sample in enumerate (test_samples) :
        type_ids_pred = te_entity_vecs[j].nonzero ()[0]
        labels = utils.get_full_types ([gres.type_vocab[tid] for tid in type_ids_pred])
        if len (labels) == 0 :
            cnt += 1
            labels = ['NAN']
        pred_labels_dict_te[sample.mention_id] = labels

    print ("dev missing labels # = ", cnt)
    cnt = 0
    for j, sample in enumerate (dev_samples) :
        type_ids_pred = dv_entity_vecs[j].nonzero ()[0]
        labels = utils.get_full_types ([gres.type_vocab[tid] for tid in type_ids_pred])
        if len (labels) == 0 :
            cnt += 1
            labels = ['NAN']
        pred_labels_dict_dv[sample.mention_id] = labels
    print ("dev missing labels # = ", cnt)

    def _eval(true_labels_dict, pred_labels_dict) :
        strict_acc = utils.strict_acc (true_labels_dict, pred_labels_dict)
        partial_acc = utils.partial_acc (true_labels_dict, pred_labels_dict)
        maf1 = utils.macrof1 (true_labels_dict, pred_labels_dict)
        mif1 = utils.microf1 (true_labels_dict, pred_labels_dict)
        return strict_acc, partial_acc, maf1, mif1

    print ("Test")
    hit = {}
    miss = {}
    for k, v in pred_labels_dict_te.items () :
        pred = pred_labels_dict_te[k]
        tot = len (v)
        cnt = 0
        for p in pred :
            if p in test_true_labels_dict[k] :
                cnt += 1
                hit[p] = hit.get (p, 0) + 1
            else :
                miss[p] = miss.get (p, 0) + 1

        # print ("{}:{} {}/{} {}".format (k, one_type, cnt, tot, te_el_probs_dict[k]))
    for k, v in hit.items () :
        print ("{}: {}/{}".format (k, v, v + miss.get (k, 0)))

    print ("Dev")
    hit = {}
    miss = {}
    for k, v in pred_labels_dict_dv.items () :
        pred = pred_labels_dict_dv[k]
        tot = len (v)
        cnt = 0
        for p in pred :
            if p in dev_true_labels_dict[k] :
                cnt += 1
                hit[p] = hit.get (p, 0) + 1
            else :
                miss[p] = miss.get (p, 0) + 1
        # print ("{}: {} {}/{} {}".format (k, one_type, cnt, tot, dv_el_probs_dict[k]))
    for k, v in hit.items () :
        print ("{}: {}/{}".format (k, v, v + miss.get (k, 0)))
    strict_acct, partial_acct, maf1t, mif1t = _eval (test_true_labels_dict, pred_labels_dict_te)
    strict_accv, partial_accv, maf1v, mif1v = _eval (dev_true_labels_dict, pred_labels_dict_dv)
    print ("strict_accv, partial_accv, maf1v, mif1v")
    print ("test:", end='')
    print (strict_acct, partial_acct, maf1t, mif1t)
    print ("dev:", end='')
    print (strict_accv, partial_accv, maf1v, mif1v)


def _get_feature_from_batch_samples(batch_samples) :
    feats = []
    feat_set = {}
    for sample in batch_samples :
        feat = MentionFeat.features (sample)
        feats.append (feat)
    return feats

def train_fetel(args, writer, device, gres: exputils.GlobalRes, el_entityvec: ELDirectEntityVec, train_samples_pkl,
                dev_samples_pkl, test_mentions_file, test_sents_file, test_noel_preds_file, type_embed_dim,
                context_lstm_hidden_dim, learning_rate, batch_size, n_iter, dropout, rand_per, per_penalty,
                use_mlp=False, pred_mlp_hdim=None, save_model_file=None, nil_rate=0.5,
                single_type_path=False, stack_lstm=False, concat_lstm=False, results_file=None, feat_dim=16) :
    logging.info('result_file={}'.format(results_file))
    logging.info(
        'type_embed_dim={} cxt_lstm_hidden_dim={} pmlp_hdim={} nil_rate={} single_type_path={}'.format(
            type_embed_dim, context_lstm_hidden_dim, pred_mlp_hdim, nil_rate, single_type_path))
    logging.info('rand_per={} per_pen={}'.format(rand_per, per_penalty))
    logging.info('stack_lstm={} cat_lstm={}'.format(stack_lstm, concat_lstm))

    if stack_lstm:
        print ("Use [{}] GPUs".format (torch.cuda.device_count ()))
        model = NoName(
            device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer, context_lstm_hidden_dim,
            feat_set=MentionFeat.get_feat_set (),
            type_embed_dim=type_embed_dim, dropout=dropout, use_mlp=use_mlp, mlp_hidden_dim=pred_mlp_hdim,
            concat_lstm=concat_lstm, copy=args.copy, feat_emb_dim=feat_dim)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    else:
        model = None
    if device.type == 'cuda':
        model = model.cuda ()
    if args.resume != "" :
        logging.info ("resume from {}".format (args.resume))
        model.load_state_dict (torch.load (args.resume))

    # 每个sample都是一个长度为7的tuple：
    train_samples = datautils.load_pickle_data(train_samples_pkl)
    train_size = len(train_samples)
    dev_samples = datautils.load_pickle_data(dev_samples_pkl)
    # batch_samples：针对每一个mention，context的token sequence id，包括parent的full types
    dev_samples = anchor_samples_to_model_samples(dev_samples, gres.mention_token_id, gres.parent_type_ids_dict)  # type: List[LabeledModelSample]
    dev_feats = _get_feature_from_batch_samples (dev_samples)
    start = time.time()
    print ("train_samples_with_label takes {}s".format ((time.time () - start) / 1000))
    lr_gamma = 0.7
    eval_batch_size = 32
    logging.info('{}'.format(model.__class__.__name__))
    dev_true_labels_dict = {s.mention_id: [gres.type_vocab[l] for l in s.labels] for s in dev_samples}
    # dev_entity_vecs： (2000, 128)
    dev_entity_vecs, dev_el_sgns, dev_el_probs = __get_entity_vecs_for_samples(el_entityvec, dev_samples, None)
    test_samples = model_samples_from_json(gres.token_id_dict,
                                           gres.unknown_token_id,
                                           gres.mention_token_id,
                                           gres.type_id_dict,
                                           test_mentions_file,
                                           test_sents_file)
    te_feats = _get_feature_from_batch_samples (test_samples)
    test_noel_pred_results = datautils.read_pred_results_file(test_noel_preds_file)

    test_mentions = datautils.read_json_objs(test_mentions_file)
    test_entity_vecs, test_el_sgns, test_el_probs = __get_entity_vecs_for_mentions(el_entityvec,
                                                                                   test_mentions,
                                                                                   test_noel_pred_results,
                                                                                   gres.n_types)

    test_true_labels_dict = {m['mention_id']: m['labels'] for m in test_mentions} if (
            'labels' in next(iter(test_mentions))) else None

    person_type_id = gres.type_id_dict.get('/person')
    l2_person_type_ids, person_loss_vec = None, None
    if person_type_id is not None:
        l2_person_type_ids = __get_l2_person_type_ids(gres.type_vocab)
        person_loss_vec = exputils.get_person_type_loss_vec (l2_person_type_ids, gres.n_types, per_penalty, device)

    dev_results_file = None
    n_batches = (len(train_samples) + batch_size - 1) // batch_size
    optimizer = torch.optim.Adam (filter (lambda p : p.requires_grad, model.parameters ()), lr=learning_rate)
    nelement = sum ([p.nelement () for p in model.parameters () if not p.requires_grad])
    logging.info ("number of training params is {}".format (nelement))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batches, gamma=lr_gamma)
    losses = list()
    best_dev_acc = -1
    logging.info('{} steps, {} steps per iter, lr_decay={}, start training ...'.format(
        n_iter * n_batches, n_batches, lr_gamma))
    step = 0
    n_steps = n_iter * n_batches
    if not args.eval :
        while step < n_steps :
            batch_idx = step % n_batches
            batch_beg, batch_end = batch_idx * batch_size, min ((batch_idx + 1) * batch_size, len (train_samples))

            batch_samples = anchor_samples_to_model_samples (
                train_samples[batch_beg :batch_end], gres.mention_token_id, gres.parent_type_ids_dict)
            feats = _get_feature_from_batch_samples (batch_samples)
            # entity_vecs should be the linked types
            if rand_per :
                entity_vecs, el_sgns, el_probs = __get_entity_vecs_for_samples (el_entityvec, batch_samples, None, True,
                                                                                person_type_id, l2_person_type_ids,
                                                                                gres.type_vocab)
            else :
                entity_vecs, el_sgns, el_probs = __get_entity_vecs_for_samples (el_entityvec, batch_samples, None, True)

            use_entity_vecs = True
            model.train ()

            (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, y_true
             ) = exputils.get_mstr_cxt_label_batch_input (device, gres.n_types, batch_samples)

            if use_entity_vecs :
                for i in range (entity_vecs.shape[0]) :
                    if np.random.uniform () < nil_rate :
                        entity_vecs[i] = np.zeros (entity_vecs.shape[1], np.float32)
                el_probs = torch.tensor (el_probs, dtype=torch.float32, device=device)
                entity_vecs = torch.tensor (entity_vecs, dtype=torch.float32, device=device)
            else :
                entity_vecs = None
            logits = model (context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs, feats)
            loss = model.get_loss (y_true, logits, person_loss_vec=person_loss_vec)
            scheduler.step ()
            optimizer.zero_grad ()
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ (model.parameters (), 10.0, float ('inf'))
            optimizer.step ()
            losses.append (loss.data.cpu ().numpy ())

            step += 1
            if step % 1000 == 0 :
                acc_tr, pacc_tr = -1, -1
                acc_v, pacc_v, _, _, dev_results = eval_fetel (args,
                                                               device, gres, model, dev_samples, dev_entity_vecs,
                                                               dev_el_probs, batch_size=eval_batch_size,
                                                               use_entity_vecs=use_entity_vecs,
                                                               single_type_path=single_type_path,
                                                               true_labels_dict=dev_true_labels_dict, test=False,
                                                               feats=dev_feats)

                acc_t, _, maf1, mif1, test_results = eval_fetel (args,
                                                                 device, gres, model, test_samples, test_entity_vecs,
                                                                 test_el_probs, batch_size=eval_batch_size,
                                                                 use_entity_vecs=use_entity_vecs,
                                                                 single_type_path=single_type_path,
                                                                 true_labels_dict=test_true_labels_dict, test=True,
                                                                 feats=te_feats)

                best_tag = '*' if acc_v > best_dev_acc else ''
                logging.info (
                    'i={} l={:.4f} acctr = {:.4f}  pacctr = {:.4f} accv={:.4f} paccv={:.4f} acct={:.4f} maf1={:.4f} mif1={:.4f}{}'.format (
                        step, sum (losses), acc_tr, pacc_tr, acc_v, pacc_v, acc_t, maf1, mif1, best_tag))
                writer.add_scalar ("acc_tr", acc_tr)
                writer.add_scalar ("pacc_tr", pacc_tr)
                writer.add_scalar ("accv", acc_v)
                writer.add_scalar ("paccv", pacc_v)
                writer.add_scalar ("acct", acc_t)
                writer.add_scalar ("maf1", maf1)
                writer.add_scalar ("mif1", mif1)

                if acc_v > best_dev_acc and save_model_file :
                    torch.save (model.state_dict (), "{}.{}".format (save_model_file, step))
                    logging.info ('model saved to {}'.format ("{}.{}".format (save_model_file, step)))

                if dev_results_file is not None and acc_v > best_dev_acc :
                    datautils.save_json_objs (dev_results, dev_results_file)
                    logging.info ('dev reuslts saved {}'.format (dev_results_file))
                if results_file is not None and acc_v > best_dev_acc :
                    datautils.save_json_objs (test_results, results_file)
                    logging.info ('test reuslts saved {}'.format (results_file))

                if acc_v > best_dev_acc :
                    best_dev_acc = acc_v
                losses = list ()
    else :
        acc_tr, pacc_tr = -1, -1
        use_entity_vecs = True
        acc_v, pacc_v, _, _, dev_results = eval_fetel (args,
                                                       device, gres, model, dev_samples, dev_entity_vecs,
                                                       dev_el_probs, batch_size=eval_batch_size,
                                                       use_entity_vecs=use_entity_vecs,
                                                       single_type_path=single_type_path,
                                                       true_labels_dict=dev_true_labels_dict, test=False,
                                                       feats=dev_feats)

        acc_t, _, maf1, mif1, test_results = eval_fetel (args,
                                                         device, gres, model, test_samples, test_entity_vecs,
                                                         test_el_probs, batch_size=eval_batch_size,
                                                         use_entity_vecs=use_entity_vecs,
                                                         single_type_path=single_type_path,
                                                         true_labels_dict=test_true_labels_dict, test=True,
                                                         feats=te_feats)

        best_tag = '*' if acc_v > best_dev_acc else ''
        logging.info (
            'i={} l={:.4f} acctr = {:.4f}  pacctr = {:.4f} accv={:.4f} paccv={:.4f} acct={:.4f} maf1={:.4f} mif1={:.4f}{}'.format (
                step, sum (losses), acc_tr, pacc_tr, acc_v, pacc_v, acc_t, maf1, mif1, best_tag))


def eval_fetel(args, device, gres: exputils.GlobalRes, model, samples: List[ModelSample], entity_vecs, el_probs,
               feats=None,
               batch_size=32,
               use_entity_vecs=True,
               single_type_path=False,
               true_labels_dict=None,
               test=True, ) :
    model.eval()
    n_batches = (len(samples) + batch_size - 1) // batch_size
    pred_labels_dict = dict()
    result_objs = list()
    for i in range(n_batches):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]
        (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs
         ) = exputils.get_mstr_cxt_batch_input(batch_samples)
        entity_vecs_batch, el_probs_batch = None, None
        if use_entity_vecs:
            # entity_vecs, el_sgns = __get_entity_vecs_for_samples(el_entityvec, batch_samples, noel_pred_results)
            entity_vecs_batch = torch.tensor(entity_vecs[batch_beg:batch_end], dtype=torch.float32,
                                             device=device)
            # el_sgns_batch = torch.tensor(el_sgns[batch_beg:batch_end], dtype=torch.float32, device=device)
            el_probs_batch = torch.tensor (el_probs[batch_beg :batch_end], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs,
                           entity_vecs_batch, el_probs_batch, feats)

        if single_type_path:
            preds = model.inference(logits)
        else:
            preds = model.inference_full(logits, extra_label_thres=0.0)
        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(batch_samples, preds, logits.data.cpu().numpy())):
            labels = utils.get_full_types([gres.type_vocab[tid] for tid in type_ids_pred])
            pred_labels_dict[sample.mention_id] = labels
            result_objs.append({'mention_id': sample.mention_id, 'labels': labels,
                                'logits': [float(v) for v in sample_logits]})

    strict_acc, partial_acc, maf1, mif1 = 0, 0, 0, 0
    if true_labels_dict is not None:
        strict_acc = utils.strict_acc (true_labels_dict, pred_labels_dict, test)
        partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
        maf1 = utils.macrof1(true_labels_dict, pred_labels_dict)
        mif1 = utils.microf1(true_labels_dict, pred_labels_dict)

    return strict_acc, partial_acc, maf1, mif1, result_objs
