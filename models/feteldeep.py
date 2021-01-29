import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import modelutils


def inference_labels_full(l1_type_indices, child_type_vecs, scores, extra_label_thres=0.5):
    label_preds_main = inference_labels(l1_type_indices, child_type_vecs, scores)
    label_preds = list()
    for i in range(len(scores)):
        extra_idxs = np.argwhere(scores[i] > extra_label_thres).squeeze(axis=1)
        label_preds.append(list(set(label_preds_main[i] + list(extra_idxs))))
    return label_preds


def inference_labels(l1_type_indices, child_type_vecs, scores, multilabel) :
    l1_type_scores = scores[:, l1_type_indices]
    tmp_indices = np.argmax(l1_type_scores, axis=1)
    max_l1_indices = l1_type_indices[tmp_indices]
    l2_scores = child_type_vecs[max_l1_indices] * scores
    max_l2_indices = np.argmax(l2_scores, axis=1)
    # labels_pred = np.zeros(scores.shape[0], np.int32)
    labels_pred = list()
    for i, (l1_idx, l2_idx) in enumerate(zip(max_l1_indices, max_l2_indices)):
        # labels_pred[i] = l2_idx if l2_scores[i][l2_idx] > 1e-4 else l1_idx
        labels_pred.append([l2_idx] if l2_scores[i][l2_idx] > 1e-4 else [l1_idx])
    return labels_pred


class BaseResModel(nn.Module):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding,
                 context_lstm_hidden_dim, type_embed_dim, dropout=0.5, concat_lstm=False):
        super(BaseResModel, self).__init__()
        self.device = device
        self.context_lstm_hidden_dim = context_lstm_hidden_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.type_vocab, self.type_id_dict = type_vocab, type_id_dict
        self.l1_type_indices, self.l1_type_vec, self.child_type_vecs = modelutils.build_hierarchy_vecs(
            self.type_vocab, self.type_id_dict)
        self.n_types = len(self.type_vocab)
        self.type_embed_dim = type_embed_dim
        self.type_embeddings = torch.tensor(np.random.normal(
            scale=0.01, size=(type_embed_dim, self.n_types)).astype(np.float32),
                                            device=self.device, requires_grad=True)
        self.type_embeddings = nn.Parameter(self.type_embeddings)

        self.word_vec_dim = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer

        self.concat_lstm = concat_lstm
        self.context_lstm1 = nn.LSTM(input_size=self.word_vec_dim, hidden_size=self.context_lstm_hidden_dim,
                                     bidirectional=True)
        self.context_hidden1 = None

        self.context_lstm2 = nn.LSTM(input_size=self.context_lstm_hidden_dim * 2,
                                     hidden_size=self.context_lstm_hidden_dim, bidirectional=True)
        t_layer = nn.TransformerEncoderLayer (d_model=self.word_vec_dim, nhead=4, dim_feedforward=2 * self.word_vec_dim)
        self.t_layer = nn.TransformerEncoder (num_layers=2, encoder_layer=t_layer)
        self.context_hidden2 = None

    def init_context_hidden(self, batch_size):
        return modelutils.init_lstm_hidden(self.device, batch_size, self.context_lstm_hidden_dim, True)

    def get_context_transformer_output(self, word_id_seqs, lens, mention_tok_idxs, batch_size) :
        x = self.embedding_layer (word_id_seqs)
        B = x.size (0)
        T = x.size (1)
        mask = torch.zeros ((B, T), device=self.device).bool ()
        for i in range (B) :
            length = lens[i]
            mask[i, length :] = True
        x = self.t_layer (x.transpose (0, 1), src_key_padding_mask=mask).transpose (0, 1)
        lstm_output_r = x[list (range (batch_size)), mention_tok_idxs, :]  # 找到mention的位置的embedding
        return lstm_output_r

    def get_context_lstm_output(self, word_id_seqs, lens, mention_tok_idxs, batch_size):
        # word_id_seqs: (B, max_len), mention_tok_idxs: (B,)
        self.context_hidden1 = self.init_context_hidden(batch_size)
        self.context_hidden2 = self.init_context_hidden(batch_size)

        x = self.embedding_layer(word_id_seqs)
        x = torch.nn.utils.rnn.pack_padded_sequence (x, lens.cpu () if torch.version.__version__.startswith (
            '1.7') else lens,
                                                     batch_first=True)
        lstm_output1, self.context_hidden1 = self.context_lstm1(x, self.context_hidden1)
        # lstm_output1 = self.dropout_layer(lstm_output1)
        lstm_output2, self.context_hidden2 = self.context_lstm2(lstm_output1, self.context_hidden2)

        lstm_output1, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output1, batch_first=True) #(B, T, D)
        lstm_output2, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output2, batch_first=True)
        if self.concat_lstm:
            lstm_output = torch.cat((lstm_output1, lstm_output2), dim=2)
        else:
            lstm_output = lstm_output1 + lstm_output2

        lstm_output_r = lstm_output[list(range(batch_size)), mention_tok_idxs, :] # 找到mention的位置的embedding
        return lstm_output_r

    def get_loss(self, true_type_vecs, scores, margin=1.0, person_loss_vec=None) :
        tmp1 = torch.sum (true_type_vecs * F.relu (margin - scores), dim=1)
        tmp2 = (1 - true_type_vecs) * F.relu (margin + scores)
        if person_loss_vec is not None :
            tmp2 *= person_loss_vec.view (-1, self.n_types)
        tmp2 = torch.sum (tmp2, dim=1)
        loss = torch.mean (torch.add (tmp1, tmp2))
        return loss

    def inference(self, scores, is_torch_tensor=True, multilabel=True) :
        if is_torch_tensor:
            scores = scores.data.cpu().numpy()
        return inference_labels (self.l1_type_indices, self.child_type_vecs, scores, multilabel)

    def inference_full(self, logits, extra_label_thres=0.5, is_torch_tensor=True):
        if is_torch_tensor:
            logits = logits.data.cpu().numpy()
        return inference_labels_full(self.l1_type_indices, self.child_type_vecs, logits, extra_label_thres)

    def forward(self, *input_args):
        raise NotImplementedError

class CopyMode(nn.Module):
    def __init__(self, input_size, out_size, mlp_hidden_dim=None, dp=0.5) :
        super().__init__()
        layers = []

        mlp_hidden_dim = input_size // 2 if mlp_hidden_dim is None else mlp_hidden_dim
        layers.append (nn.Linear (input_size, mlp_hidden_dim, bias=False))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Linear (mlp_hidden_dim, mlp_hidden_dim, bias=False))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Linear (mlp_hidden_dim, out_size, bias=False))

        self.fc = nn.Sequential (*layers)

    def forward(self, x, entity_vecs, type_emb, topk=2) :
        """x: (256, 800)
           entity_vecs: (B, n_type)
           type_emb: (emb_size, n_type)
        """
        x = self.fc (x)  # (B, 1, D) x (1, D, O)
        type_embed_dim, n_types = type_emb.size ()
        logits = torch.matmul (x.view (-1, 1, type_embed_dim),
                               type_emb.view (-1, type_embed_dim, n_types))
        logits = logits.view (-1, n_types)  # (B, O)
        logits = F.relu (logits)
        return logits * entity_vecs

class GenerationMode(nn.Module):
    def __init__(self, input_size, type_embed_dim, use_mlp=False, mlp_hidden_dim=None, dp=0.5):
        super().__init__()
        layers = []
        if not use_mlp :
            layers.append (nn.Dropout(dp))
            layers.append (nn.Linear (input_size, type_embed_dim, bias=False))
            # layers.append (nn.Tanh ())
        else :
            mlp_hidden_dim = input_size // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            layers.append (nn.Linear (input_size, mlp_hidden_dim))
            layers.append (nn.ReLU ())
            layers.append (nn.BatchNorm1d (mlp_hidden_dim))
            layers.append (nn.Dropout (dp))
            layers.append (nn.Linear (mlp_hidden_dim, mlp_hidden_dim))
            layers.append (nn.ReLU ())
            layers.append (nn.BatchNorm1d (mlp_hidden_dim))
            layers.append (nn.Dropout (dp))
            layers.append (nn.Linear (mlp_hidden_dim, type_embed_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, type_emb):
        x = self.fc(x)
        type_embed_dim, n_types = type_emb.size()
        logits = torch.matmul (x.view (-1, 1, type_embed_dim),
                               type_emb.view (-1, type_embed_dim, n_types))
        logits = logits.view (-1, n_types)
        return logits

class NoName(BaseResModel):
    """could get 76% at least"""
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim,
                 dropout=0.5,
                 use_mlp=False,
                 mlp_hidden_dim=None,
                 concat_lstm=False,
                 copy=True,
                 alpha=0.5
                 ) :
        super(NoName, self).__init__(device, type_vocab, type_id_dict, embedding_layer,
                                         context_lstm_hidden_dim, type_embed_dim, dropout, concat_lstm)
        self.use_mlp = use_mlp
        self.copy = copy
        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim
        if concat_lstm:
            linear_map_input_dim += 2 * self.context_lstm_hidden_dim
        hidden_size = 512
        layers = []
        mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
        layers.append (nn.Linear (linear_map_input_dim, mlp_hidden_dim))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Dropout (dropout))
        layers.append (nn.Linear (mlp_hidden_dim, mlp_hidden_dim))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Dropout (dropout))
        layers.append (nn.Linear (mlp_hidden_dim, hidden_size))

        self.encoder = nn.Sequential (*layers)
        self.generate_mode = nn.Linear (hidden_size, self.n_types)
        self.copy_mode = nn.Sequential (nn.Linear (self.n_types + 1, hidden_size),
                                        nn.ReLU (),
                                        nn.BatchNorm1d (hidden_size),
                                        nn.Dropout (dropout),
                                        nn.Linear (hidden_size, hidden_size),
                                        nn.ReLU (),
                                        nn.BatchNorm1d (hidden_size),
                                        nn.Dropout (dropout),
                                        nn.Linear (hidden_size, self.n_types),
                                        )
        self.alpha = alpha

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs, pos_feats) :
        """

        :param context_token_seqs: List[List[Int]], len(List) = batch_size  sent_tokens[:pos_beg] + [mention_token_id] + sent_tokens[pos_end:]
        :param mention_token_idxs: List[Int], len(List) = batch_size： mention在句子里面的starting index
        :param mstr_token_seqs: List[List[Int]], len(List) = batch_size, List里面的元素个数非常的少
        :param entity_vecs: (batch_size x 128) linking results, multihot vector
        :param el_probs: (batch_size,) linking score
        :return:
        """
        batch_size = len(context_token_seqs)
        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input(
            self.device, context_token_seqs, mention_token_idxs)

        context_lstm_output = self.get_context_lstm_output(context_token_seqs, seq_lens, mention_token_idxs, batch_size) # (B, D) or (B, 2*D)

        # step 1: context
        context_lstm_output = context_lstm_output[back_idxs] # (256, 500)

        # step 2: mention str vector
        # (256, 300)
        name_output = modelutils.get_avg_token_vecs (self.device, self.embedding_layer,
                                                     mstr_token_seqs)  # (B, D) or (B, 2*D)

        # step 3: entity_vecs: the entity linking results
        cat_output = self.dropout_layer (torch.cat ((context_lstm_output, name_output), dim=1))
        state = self.encoder (cat_output)  # (B, D)
        g = self.generate_mode (state)  # (B, type_dim)
        if self.copy :
            c = self.copy_mode (torch.cat ((entity_vecs, el_probs.unsqueeze (1)), dim=1))  # (B, D)
            c = F.relu (c)
            logits = self.alpha * c + (1 - self.alpha) * g
        else :
            logits = g
        logits = logits.view(-1, self.n_types)
        return logits

class NoName3 (BaseResModel) :
    """could get 76.4% at least"""

    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim,
                 dropout=0.5,
                 use_mlp=False,
                 mlp_hidden_dim=None,
                 concat_lstm=False,
                 copy=True, alpha=0.5) :
        super (NoName3, self).__init__ (device, type_vocab, type_id_dict, embedding_layer,
                                        context_lstm_hidden_dim, type_embed_dim, dropout, concat_lstm)
        self.use_mlp = use_mlp
        self.copy = copy
        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim
        # linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types + 1
        if concat_lstm :
            linear_map_input_dim += 2 * self.context_lstm_hidden_dim
        hidden_size = 512
        layers = []
        mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
        layers.append (nn.Linear (linear_map_input_dim, mlp_hidden_dim))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Dropout (dropout))
        layers.append (nn.Linear (mlp_hidden_dim, mlp_hidden_dim))
        layers.append (nn.ReLU ())
        layers.append (nn.BatchNorm1d (mlp_hidden_dim))
        layers.append (nn.Dropout (dropout))
        layers.append (nn.Linear (mlp_hidden_dim, hidden_size))

        self.encoder = nn.Sequential (*layers)
        if self.copy :
            self.alpha = nn.Sequential (nn.Linear (1, 1), nn.Sigmoid ())
        self.generate_mode = nn.Linear (hidden_size, self.n_types)
        self.copy_mode = nn.Sequential (nn.Linear (self.n_types + 1, hidden_size),
                                        nn.ReLU (),
                                        nn.BatchNorm1d (hidden_size),
                                        nn.Dropout (dropout),
                                        nn.Linear (hidden_size, hidden_size),
                                        nn.ReLU (),
                                        nn.BatchNorm1d (hidden_size),
                                        nn.Dropout (dropout),
                                        nn.Linear (hidden_size, self.n_types),
                                        )
        self.alpha = alpha

    def _load_type_emb(self, path, type_id_dict) :
        type2vec = {}
        with open (path, 'r') as f :
            line = f.readline ()
            n_type = int (line.split ()[0])
            dim = int (line.split ()[1])
            for line in f :
                segs = line.strip ().split ()
                if len (segs) == 2 :
                    continue
                typename = segs[0]
                vect = np.array (segs[1 :]).astype (np.float)
                type2vec[typename] = vect
        arr = np.zeros ((dim, n_type))
        for name, vec in type2vec.items () :
            arr[:, type_id_dict[name]] = vec
        return arr

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs, pos_feats) :
        """

        :param context_token_seqs: List[List[Int]], len(List) = batch_size  sent_tokens[:pos_beg] + [mention_token_id] + sent_tokens[pos_end:]
        :param mention_token_idxs: List[Int], len(List) = batch_size： mention在句子里面的starting index
        :param mstr_token_seqs: List[List[Int]], len(List) = batch_size, List里面的元素个数非常的少
        :param entity_vecs: (batch_size x 128) linking results, multihot vector
        :param el_probs: (batch_size,) linking score
        :return:
        """
        batch_size = len (context_token_seqs)
        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input (
            self.device, context_token_seqs, mention_token_idxs)

        context_lstm_output = self.get_context_lstm_output (context_token_seqs, seq_lens, mention_token_idxs,
                                                            batch_size)  # (B, D) or (B, 2*D)

        # step 1: context
        context_lstm_output = context_lstm_output[back_idxs]  # (256, 500)

        # step 2: mention str vector
        # (256, 300)
        name_output = modelutils.get_avg_token_vecs (self.device, self.embedding_layer,
                                                     mstr_token_seqs)  # (B, D) or (B, 2*D)

        # step 3: entity_vecs: the entity linking results
        cat_output = self.dropout_layer (torch.cat ((context_lstm_output, name_output), dim=1))
        state = self.encoder (cat_output)  # (B, D)
        g = self.generate_mode (state)  # (B, type_dim)
        if self.copy :
            c = self.copy_mode (torch.cat ((entity_vecs, el_probs.unsqueeze (1)), dim=1))  # (B, D)
            return g.view (-1, self.n_types), c.view (-1, self.n_types)
        else :
            return g.view (-1, self.n_types)

