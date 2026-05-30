"""Models for Stage 3.

``KernelKGGPT`` reuses the KernelGAT equations verbatim (21 Gaussian kernels,
node kernel, sentence-level GAT, log-aggregated verification probability). The
only differences from the reference KernelGAT implementation are:

1. Input nodes are ``[CLS] claim [SEP] triple [SEP]`` (triple = "head rel tail")
   instead of ``[CLS] claim [SEP] wiki_title evidence [SEP]``.
2. Kernel mu/sigma are registered as buffers so the model runs on CPU as
   well as CUDA (the reference impl hard-codes ``.cuda()`` calls).
3. NULL padding nodes are masked out of the softmaxes so a variable number
   of retrieved triples can be padded up to ``max_nodes``.
4. ``num_labels = 2`` (FactKG: True / False).

``BertConcatBaseline`` is a deliberately simple *supervised* baseline used for
a fair comparison: it sees the exact same triple pool but drops the kernel /
GAT machinery (mean-pools the per-node [CLS] vectors, then a linear head). If
``KernelKGGPT`` beats this baseline, the gain is attributable to the KernelGAT
architecture rather than merely to supervised fine-tuning on FactKG labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


def kernel_mus(n_kernels: int):
    l_mu = [1.0]
    if n_kernels == 1:
        return l_mu
    bin_size = 2.0 / (n_kernels - 1)
    l_mu.append(1 - bin_size / 2)
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels: int):
    l_sigma = [0.001]
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma


class KernelKGGPT(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_kernels: int = 21,
        num_labels: int = 2,
        max_nodes: int = 10,
        max_seq_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.num_kernels = num_kernels
        self.num_labels = num_labels
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Kernel buffers — shape (1, 1, 1, K) like KernelGAT
        mu = torch.FloatTensor(kernel_mus(num_kernels)).view(1, 1, 1, num_kernels)
        sigma = torch.FloatTensor(kernel_sigmas(num_kernels)).view(1, 1, 1, num_kernels)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        # KernelGAT projections (names mirror the reference impl)
        self.proj_select = nn.Linear(num_kernels, 1)
        self.proj_att = nn.Linear(num_kernels, 1)
        self.proj_inference_de = nn.Linear(self.hidden_dim * 2, num_labels)
        self.proj_gat = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------ #
    # Kernel ops                                                          #
    # ------------------------------------------------------------------ #
    def _kernel_pool_node(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.unsqueeze(-1)
        attn_d = attn_d.unsqueeze(1).unsqueeze(-1)
        sim = torch.bmm(q_embed, d_embed.transpose(1, 2)).unsqueeze(-1)
        pooling = torch.exp(-((sim - self.mu) ** 2) / (self.sigma ** 2) / 2) * attn_d
        pooling_sum = pooling.sum(2)
        log_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q
        log_sum = log_sum.sum(1) / (attn_q.sum(1) + 1e-10)
        return self.proj_select(log_sum)

    def _kernel_pool_token(self, q_embed, d_embed, attn_q, attn_d):
        attn_d = attn_d.unsqueeze(1).unsqueeze(-1)
        sim = torch.bmm(q_embed, d_embed.transpose(1, 2)).unsqueeze(-1)
        pooling = torch.exp(-((sim - self.mu) ** 2) / (self.sigma ** 2) / 2) * attn_d
        log_sum = torch.log(torch.clamp(pooling.sum(2), min=1e-10))
        log_sum = self.proj_att(log_sum).squeeze(-1)
        log_sum = log_sum.masked_fill((1 - attn_q).bool(), -1e4)
        return F.softmax(log_sum, dim=1)

    # ------------------------------------------------------------------ #
    # Sentence-level GAT attention                                        #
    # ------------------------------------------------------------------ #
    def _self_attention(self, inputs, inputs_hiddens, mask_text, idx, is_null):
        # Matches KernelGAT: the token-level kernel attention uses the FULL
        # text mask (claim + triple) for both the query and the anchor, i.e.
        # self_attention(inputs, inputs_hiddens, mask_text, mask_text, i).
        # Only the denoised branch is kept — the non-denoised ``outputs`` is
        # computed-then-discarded in the reference impl, so we drop it.
        B, N, L, H = inputs_hiddens.shape

        own_hidden = inputs_hiddens[:, idx : idx + 1].expand(-1, N, -1, -1)
        own_mask = mask_text[:, idx : idx + 1].expand(-1, N, -1)
        own_input = inputs[:, idx : idx + 1].expand(-1, N, -1)

        own_norm = F.normalize(own_hidden, p=2, dim=-1)
        ev_norm = F.normalize(inputs_hiddens, p=2, dim=-1)
        att = self._kernel_pool_token(
            ev_norm.reshape(-1, L, H),
            own_norm.reshape(-1, L, H),
            mask_text.reshape(-1, L),
            own_mask.reshape(-1, L),
        )
        att = att.view(B, N, L, 1)
        denoise = (att * inputs_hiddens).sum(dim=2)

        score_de = self.proj_gat(torch.cat([own_input, denoise], dim=-1))
        if is_null is not None:
            score_de = score_de.masked_fill(is_null.unsqueeze(-1), -1e4)
        w_de = F.softmax(score_de, dim=1)
        out_de = (denoise * w_de).sum(dim=1)
        return out_de

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, token_type_ids, is_null):
        B, N, L = input_ids.shape
        flat_ids = input_ids.view(-1, L)
        flat_mask = attention_mask.view(-1, L)
        flat_seg = token_type_ids.view(-1, L)

        bert_out = self.bert(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            token_type_ids=flat_seg,
        )
        hidden = self.dropout(bert_out.last_hidden_state)
        pooled = bert_out.pooler_output
        H = hidden.size(-1)

        mask_text = flat_mask.float().clone()
        mask_text[:, 0] = 0.0
        mask_claim = (1.0 - flat_seg.float()) * mask_text
        mask_evidence = flat_seg.float() * mask_text

        hidden_norm = F.normalize(hidden, p=2, dim=2)
        node_score = self._kernel_pool_node(
            hidden_norm, hidden_norm, mask_claim, mask_evidence
        )
        node_score = node_score.view(B, N, 1).masked_fill(is_null.unsqueeze(-1), -1e4)
        select_prob = F.softmax(node_score, dim=1)

        inputs = pooled.view(B, N, H)
        inputs_hiddens = hidden.view(B, N, L, H)
        mask_text_3d = mask_text.view(B, N, L)

        outputs_de = []
        for i in range(N):
            out_de = self._self_attention(
                inputs, inputs_hiddens, mask_text_3d, idx=i, is_null=is_null,
            )
            outputs_de.append(out_de)
        outputs_de = torch.stack(outputs_de, dim=1)

        feat = torch.cat([inputs, outputs_de], dim=-1)
        per_node_logits = self.proj_inference_de(feat)
        per_node_prob = F.softmax(per_node_logits, dim=-1)

        agg_prob = (select_prob * per_node_prob).sum(dim=1)
        agg_prob = torch.clamp(agg_prob, min=1e-10)
        logits = torch.log(agg_prob)

        return {
            "logits": logits,
            "per_node_pred": per_node_prob,
            "node_probs": select_prob.squeeze(-1),
        }


class BertConcatBaseline(nn.Module):
    """Fair supervised baseline — same triples, no kernel/GAT machinery.

    Encodes each ``[CLS] claim [SEP] triple [SEP]`` node, mean-pools the node
    [CLS] vectors over non-NULL nodes, then a linear classification head.
    Returns log-probabilities so it plugs into the same ``NLLLoss`` head.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        **kwargs,  # accept/ignore kernel-specific kwargs for a uniform factory
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, is_null):
        B, N, L = input_ids.shape
        bert_out = self.bert(
            input_ids=input_ids.view(-1, L),
            attention_mask=attention_mask.view(-1, L),
            token_type_ids=token_type_ids.view(-1, L),
        )
        cls = bert_out.pooler_output.view(B, N, self.hidden_dim)

        # Mean-pool over non-NULL nodes (avoid dividing by zero if all NULL).
        valid = (~is_null).float().unsqueeze(-1)              # (B, N, 1)
        denom = valid.sum(dim=1).clamp(min=1.0)               # (B, 1)
        pooled = (cls * valid).sum(dim=1) / denom             # (B, H)

        logits = self.classifier(self.dropout(pooled))
        log_probs = F.log_softmax(logits, dim=-1)
        return {
            "logits": log_probs,
            "per_node_pred": None,
            "node_probs": None,
        }


def build_model(config):
    """Factory selecting the Stage-3 model from ``config['model_type']``."""
    model_type = config.get("model_type", "kernel")
    if model_type == "kernel":
        return KernelKGGPT(
            bert_model_name=config["bert_model"],
            num_kernels=config["num_kernels"],
            num_labels=config.get("num_labels", 2),
            max_nodes=config["max_nodes"],
            max_seq_len=config["max_seq_len"],
            dropout=config.get("dropout", 0.1),
        )
    elif model_type == "concat_baseline":
        return BertConcatBaseline(
            bert_model_name=config["bert_model"],
            num_labels=config.get("num_labels", 2),
            dropout=config.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown model_type: {model_type}")
