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


def inference_labels(l1_type_indices, child_type_vecs, scores):
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
        self.context_hidden2 = None

    def init_context_hidden(self, batch_size):
        return modelutils.init_lstm_hidden(self.device, batch_size, self.context_lstm_hidden_dim, True)

    def get_context_lstm_output(self, word_id_seqs, lens, mention_tok_idxs, batch_size):
        # word_id_seqs: (B, max_len), mention_tok_idxs: (B,)
        self.context_hidden1 = self.init_context_hidden(batch_size)
        self.context_hidden2 = self.init_context_hidden(batch_size)

        x = self.embedding_layer(word_id_seqs)
        # x = F.dropout(x, self.dropout, training)
        x = torch.nn.utils.rnn.pack_padded_sequence (x, lens.cpu () if torch.version.startswith ('1.7.1') else lens,
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
        # lstm_output_r = F.dropout(lstm_output_r, self.dropout, training)
        return lstm_output_r

    def get_loss(self, true_type_vecs, scores, margin=1.0, person_loss_vec=None):
        tmp1 = torch.sum(true_type_vecs * F.relu(margin - scores), dim=1)
        # tmp2 = torch.sum((1 - true_type_vecs) * F.relu(margin + scores), dim=1)
        tmp2 = (1 - true_type_vecs) * F.relu(margin + scores)
        if person_loss_vec is not None:
            tmp2 *= person_loss_vec.view(-1, self.n_types)
        tmp2 = torch.sum(tmp2, dim=1)
        loss = torch.mean(torch.add(tmp1, tmp2))
        return loss

    # def get_loss(self, true_type_vecs, scores):
    #     return F.binary_cross_entropy_with_logits(scores, true_type_vecs)

    def inference(self, scores, is_torch_tensor=True):
        if is_torch_tensor:
            scores = scores.data.cpu().numpy()
        return inference_labels(self.l1_type_indices, self.child_type_vecs, scores)

    def inference_full(self, logits, extra_label_thres=0.5, is_torch_tensor=True):
        if is_torch_tensor:
            logits = logits.data.cpu().numpy()
        return inference_labels_full(self.l1_type_indices, self.child_type_vecs, logits, extra_label_thres)

    def forward(self, *input_args):
        raise NotImplementedError


class FETELStack(BaseResModel):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, mlp_hidden_dim=None, concat_lstm=False):
        super(FETELStack, self).__init__(device, type_vocab, type_id_dict, embedding_layer,
                                         context_lstm_hidden_dim, type_embed_dim, dropout, concat_lstm)
        self.use_mlp = use_mlp
        # self.dropout_layer = nn.Dropout(dropout)

        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types + 1
        if concat_lstm:
            linear_map_input_dim += 2 * self.context_lstm_hidden_dim
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, type_embed_dim, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, type_embed_dim)

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs):
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
        context_lstm_output = context_lstm_output[back_idxs] #TODO: ????

        # step 2: mention str vector
        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs) # (B, D) or (B, 2*D)

        # step 3: entity_vecs: the entity linking results
        cat_output = self.dropout_layer(torch.cat((context_lstm_output, name_output, entity_vecs), dim=1))

        cat_output = torch.cat((cat_output, el_probs.view(-1, 1)), dim=1)

        if not self.use_mlp:
            mention_reps = self.linear_map(self.dropout_layer(cat_output))

        else:
            l1_output = self.linear_map1(cat_output)
            l1_output = self.lin1_bn(F.relu(l1_output))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            mention_reps = self.linear_map3(self.dropout_layer(l2_output)) # (B, self.type_embed_dim)

        logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))  # TODO: (B, 1, D) x (B, D, K)
        logits = logits.view(-1, self.n_types) #(B, n_class)
        return logits


class CopyMode(nn.Module):
    def __init__(self, input_size, out_size, use_mlp=False, mlp_hidden_dim=None, dp=0.5):
        super().__init__()
        layers = []
        if not use_mlp :
            layers.append (nn.Dropout(dp))
            layers.append (nn.Linear (input_size+1, out_size, bias=False))
            # layers.append (nn.Tanh ())
        else :
            mlp_hidden_dim = input_size // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            layers.append (nn.Linear (input_size+1, mlp_hidden_dim))
            layers.append (nn.ReLU ())
            layers.append (nn.BatchNorm1d (mlp_hidden_dim))
            layers.append (nn.Dropout (dp))
            layers.append (nn.Linear (mlp_hidden_dim, mlp_hidden_dim))
            layers.append (nn.ReLU ())
            layers.append (nn.BatchNorm1d (mlp_hidden_dim))
            layers.append (nn.Dropout (dp))
            layers.append (nn.Linear (mlp_hidden_dim, out_size))
            # layers.append (nn.Tanh ())

        self.fc = nn.Sequential (*layers)

    def forward(self, x, entity_vecs, type_emb, scores):
        """x: (256, 800)
           entity_vecs: (B, n_type)
           type_emb: (E, O)
        """
        x = torch.cat ((x, scores.view (-1, 1)), dim=1)
        x = self.fc(x)
        type_embed_dim, n_types = type_emb.size()
        logits = torch.matmul (x.view (-1, 1, type_embed_dim),
                               type_emb.view (-1, type_embed_dim, n_types))
        logits = logits.view (-1, n_types)
        return logits * entity_vecs # [-1, 1] + (0, 1) -> [-1, 2]


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
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, mlp_hidden_dim=None, concat_lstm=False, alpha=0.5):
        super(NoName, self).__init__(device, type_vocab, type_id_dict, embedding_layer,
                                         context_lstm_hidden_dim, type_embed_dim, dropout, concat_lstm)
        self.use_mlp = use_mlp
        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim
        self.alpha = alpha
        if concat_lstm:
            linear_map_input_dim += 2 * self.context_lstm_hidden_dim
        self.copy_mode = CopyMode(linear_map_input_dim, type_embed_dim, dp=dropout)
        self.generate_mode = GenerationMode(linear_map_input_dim, type_embed_dim, dp=dropout)
        self.alpha = nn.Sequential(nn.Linear(linear_map_input_dim, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_probs):
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
        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs) # (B, D) or (B, 2*D)

        # step 3: entity_vecs: the entity linking results
        cat_output = self.dropout_layer(torch.cat((context_lstm_output, name_output), dim=1))
        a = self.copy_mode(cat_output, entity_vecs, self.type_embeddings, el_probs)
        b = self.generate_mode(cat_output, self.type_embeddings)
        r = self.alpha(cat_output)
        logits = r * a + (1-r) * b
        logits = logits.view(-1, self.n_types)
        return logits