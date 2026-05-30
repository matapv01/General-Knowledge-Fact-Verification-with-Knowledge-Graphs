# KernelKG-GPT — Báo cáo Pipeline

> Hệ thống kiểm chứng sự thật (fact verification) trên **FactKG**, kết hợp
> **KG-GPT** (truy hồi bằng chứng dựa trên LLM) với **KernelGAT** (suy luận
> bằng kernel graph attention).

---

## 1. Tổng quan & Động lực

### 1.1. Bài toán

Cho một **claim** (tuyên bố) và tập **entity** xuất hiện trong claim, xác định
claim là **True** hay **False** dựa trên một Knowledge Graph (DBpedia).

```
Input:  claim = "Barack Obama was born in Hawaii"
        entity_set = ["Barack_Obama", "Hawaii"]
Output: True / False
```

### 1.2. Hai cách tiếp cận gốc

| | KG-GPT | KernelGAT |
|---|---|---|
| Bằng chứng | Triple từ KG (có cấu trúc) | Câu văn (Wikipedia) |
| Suy luận | Prompt LLM verify | Kernel graph attention (học được) |
| Bước xây dựng evidence | Rule-based `graph_extractor` | — |
| Bước verify | 1 lần gọi LLM | Neural classifier |

### 1.3. Ý tưởng kết hợp

KG-GPT có điểm yếu ở bước cuối: **xây dựng evidence graph bằng rule cứng** rồi
**verify bằng 1 prompt LLM**. KernelKG-GPT giữ nguyên phần truy hồi mạnh của
KG-GPT (Stage 1+2) và **thay phần verify bằng kiến trúc kernel reasoning của
KernelGAT** (Stage 3).

> **Nguyên tắc thiết kế:** kết hợp thuần túy — **không thêm tín hiệu nào ngoài
> hai paper gốc** (không gold-evidence supervision, không edge-type embedding).
> Điều này đảm bảo so sánh công bằng và tách bạch được đóng góp.

---

## 2. Kiến trúc tổng thể

Pipeline có **3 giai đoạn** chạy tuần tự. Điểm mấu chốt: **Stage 1+2 (KG-GPT)
KHÔNG cần train** (gọi GPT zero-shot), nhưng **Stage 3 (KernelGAT) BẮT BUỘC
phải train** — nó là mạng neural có trọng số học được, khởi tạo từ BERT
pretrained + các lớp Linear ngẫu nhiên, phải học từ label của FactKG mới
inference được.

```
┌──────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN A — PREPROCESS  (1 lần · GPT API · KHÔNG train)        │
│                                                                    │
│  FactKG claim ──► Stage 1 ──► Stage 2 ──► triple_pool ──► cache   │
│                  (LLM)        (KG+LLM)                             │
│  scripts/01_run_stage12.py  ·  cho cả train / dev / test split    │
└──────────────────────────────────────────────────────────────────┘
                              │  cache/stage12_{train,dev,test}.pt
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN B — TRAIN Stage 3  (GPU · HỌC trọng số)               │
│                                                                    │
│   train cache ─┐                                                  │
│                ├─► graph ─► KernelGAT ─► logits ─► NLL loss        │
│   label  ──────┘                              │                   │
│                          ▲                    ▼ backward           │
│                          └──── cập nhật trọng số (BERT + proj_*)   │
│                                                                    │
│   dev cache ──► eval mỗi epoch ──► early stopping ──► best.pt      │
│  scripts/02_train_stage3.py                                       │
└──────────────────────────────────────────────────────────────────┘
                              │  outputs/exp1/best.pt  (trọng số đã học)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN C — INFERENCE / EVAL  (GPU · trọng số ĐÃ CỐ ĐỊNH)      │
│                                                                    │
│  test cache ──► graph ──► KernelGAT(best.pt) ──► True/False        │
│  scripts/03_evaluate.py  ·  + breakdown theo reasoning type       │
└──────────────────────────────────────────────────────────────────┘
```

**Phần nào có tham số học được?**

| Thành phần | Trọng số | Train? |
|---|---|---|
| Stage 1+2 (GPT, prompt) | của OpenAI, frozen | ❌ zero-shot |
| Stage 3 — BERT encoder | pretrained → fine-tune | ✅ |
| Stage 3 — `proj_select`, `proj_att`, `proj_gat`, `proj_inference_de` | khởi tạo ngẫu nhiên | ✅ |

→ Nếu **bỏ qua Giai đoạn B**, các lớp `proj_*` còn ngẫu nhiên → Stage 3 dự đoán
~50% (đoán mò nhị phân). Vì vậy **không thể inference thẳng** mà chưa train.

**Lý do tách Giai đoạn A khỏi B+C:** Stage 1+2 gọi GPT API (tốn tiền, chậm, có
rate limit). Cache lại 1 lần → train lại / ablation Stage 3 thoải mái mà không
tốn thêm token GPT.

**Đánh đổi cần lưu ý:** KG-GPT gốc *không train gì cả* (zero-shot). KernelKG-GPT
đổi đặc tính đó lấy một model mạnh hơn nhưng **cần labeled data** (FactKG train).
Đây là lý do có `BertConcatBaseline` (xem §5.3, §9) — để tách bạch "lợi ích do
được train" khỏi "lợi ích do kiến trúc kernel".

---

## 3. Cấu trúc thư mục

```
kernelkg-gpt/
├── configs/default.yaml          # Toàn bộ hyperparameter
├── stage12/                      # GIAI ĐOẠN A — adapter cho KG-GPT
│   ├── adapter.py                # Orchestrate Stage 1+2, gọi GPT
│   └── triple_pool_builder.py    # Grounding + multi-hop bridging (logic KG-GPT)
├── stage3/                       # GIAI ĐOẠN B+C — KernelGAT (train + inference)
│   ├── triple_formatter.py       # Triple → chuỗi text
│   ├── graph_builder.py          # Triple pool → graph (NULL padding)
│   ├── data.py                   # Dataset + pre-tokenization
│   ├── model.py                  # KernelKGGPT + BertConcatBaseline
│   └── losses.py                 # NLL loss
├── scripts/
│   ├── 01_run_stage12.py         # Chạy Stage 1+2, lưu cache
│   ├── 02_train_stage3.py        # Train Stage 3
│   ├── 03_evaluate.py            # Eval + breakdown theo reasoning type
│   └── 04_ablation.py            # Chạy các biến thể ablation
├── utils/io_utils.py             # load/save pickle, set_seed
├── data/                         # FactKG + DBpedia (gitignored)
└── cache/                        # Output Stage 1+2 (gitignored)
```

Repo KG-GPT gốc nằm ở `../kg-gpt/` (thư mục anh em). Adapter import trực tiếp
các hàm từ đó.

---

## 4. GIAI ĐOẠN A — Truy hồi bằng chứng (Stage 1+2)

**Entry point:** [scripts/01_run_stage12.py](scripts/01_run_stage12.py)
**Lõi xử lý:** [stage12/adapter.py](stage12/adapter.py) `Stage12Adapter.process()`

### 4.1. Stage 1 — Phân rã claim (Sentence Division)

Gọi GPT với prompt `sentence_divide_prompt.txt` (giữ nguyên của KG-GPT) để tách
claim phức tạp thành các **sub-claim nguyên tử** (mỗi cái ≤ 2 entity).

```
"Obama's wife was born in Chicago"
        │ GPT
        ▼
{ "Obama spouse [wife]":      [Obama, Michelle],
  "[wife] birthPlace Chicago": [Michelle, Chicago] }
```

- Hàm parse: `claim_divider_parse_answer` (import từ KG-GPT)
- Retry 3 lần, `temperature=0.2, top_p=0.1`
- **Fallback:** nếu GPT lỗi → coi cả claim là 1 sub-sentence

### 4.2. Stage 2.1 — Ứng viên quan hệ (Relation Candidates)

Với mỗi sub-claim, gọi `relation_candidates(KG, type_dict, entities)` (import
từ KG-GPT, **Algorithm 1**) để lấy danh sách relation hợp lệ:

- Map entity → type qua `type_dict`
- Lấy relation giữa các type (type-based) hoặc trực tiếp từ KG (entity-based)
- Trả về `(candidate_relations, normalized_entities)` — **lưu ý** entity có thể
  được chuẩn hóa thành **tên type** (vd "Person")

### 4.3. Stage 2.2 — Xếp hạng top-K (LLM)

```python
if len(candidates) < top_k:
    chosen = candidates              # KG-GPT bỏ qua LLM khi quá ít ứng viên
else:
    chosen = GPT_rank(sub_text, candidates)[:top_k]   # relation_retrieval_prompt.txt
```

- Hàm parse: `retrieval_relation_parse_answer`
- Lọc relation về đúng candidate pool, fallback = K candidate đầu

### 4.4. Stage 2.3 — Xây dựng triple pool ⭐

**File:** [stage12/triple_pool_builder.py](stage12/triple_pool_builder.py)

Đây là phần **replicate trung thực logic KG-GPT** ([factkg_test.py:294-465](../kg-gpt/factkg_test.py#L294-L465)):

```
build_subclaim_triples()      ← Tạo triple ứng viên cho từng sub-claim
        │                        (forward + reverse "~rel", mọi cặp entity)
        ▼
build_triple_pool():
   ① additional (3-hop hints)  ← relation cho suy luận 3-hop
   ② grounding                 ← kiểm tra triple tồn tại trong KG
       • type entity → expand relation ra mọi tail thực
       • entity thường → verify cạnh tồn tại
   ③ cross-sub-claim bridging  ← NỐI entity giữa các sub-claim qua
                                  entity trung gian (multi-hop!)
   ④ dedup + graph_extractor   ← khử trùng + cắt tỉa (thứ tự xác định)
        │
        ▼
   [(head, rel, tail), ...]    ← cap tại max_triples=30
```

**Tại sao quan trọng:** với claim multi-hop *"vợ Obama sinh ở Chicago"*, entity
trung gian (Michelle) **không có** trong entity_set. Bước bridging khám phá và
nối nó qua KG — đây chính là điểm mạnh multi-hop mà bản đơn giản trước đó đã
đánh mất.

### 4.5. Output cache

[scripts/01_run_stage12.py](scripts/01_run_stage12.py) lưu **list** các record
(key bằng `qid`, không phải claim text → tránh trùng):

```python
{
  "qid": 42,
  "claim": "...",
  "entity_set": [...],
  "sub_sentences": [{text, entities, top_k_relations}, ...],
  "triple_pool": [(h, r, t), ...],
  "label": True/False,
  "reasoning_types": ["multi-hop", ...]   # từ FactKG, chỉ để breakdown
}
```

Đồng thời in **coverage**: kích thước pool trung bình + % claim có pool rỗng
(cảnh báo nếu > 15%, vì các claim này sẽ về graph toàn NULL ở Stage 3).

---

## 5. GIAI ĐOẠN B+C — Kernel reasoning (Stage 3: train rồi inference)

> Cùng một model `KernelKGGPT` được **train** ở Giai đoạn B (học trọng số từ
> label) rồi **inference** ở Giai đoạn C (dùng trọng số đã cố định). Phần data
> prep và forward dưới đây dùng chung cho cả hai.

### 5.1. Chuẩn bị dữ liệu

**File:** [stage3/data.py](stage3/data.py), [stage3/graph_builder.py](stage3/graph_builder.py)

```
triple_pool ──► build_graph() ──► max_nodes node (real + NULL pad)
                                        │
                                        ▼
   Mỗi node: [CLS] claim [SEP] "head rel tail" [SEP]
             (token_type: 0 cho claim, 1 cho triple)
                                        │
                                        ▼
   Pre-tokenize 1 LẦN → numpy compact (uint16 ids, uint8 mask)
   (tiết kiệm RAM, không tokenize lại mỗi epoch)
```

- `triple_formatter`: mặc định `"head rel tail"` (plain), tùy chọn
  `[H] head [R] rel [T] tail` (separators)
- **NULL node:** pad cho đủ `max_nodes`, đánh dấu `is_null=True`

### 5.2. Model — KernelKGGPT

**File:** [stage3/model.py](stage3/model.py)

Tái hiện **nguyên các phương trình KernelGAT** (21 Gaussian kernel). Forward gồm
5 bước:

```
input: (B, N, L)  — B claim, mỗi claim N node, mỗi node L token
   │
   ▼
① BERT ENCODER
   Encode tất cả B×N node → hidden (B*N, L, 768), pooled (B*N, 768)
   │
   ▼
② NODE KERNEL  (_kernel_pool_node)
   Mỗi node: so khớp token claim ↔ token triple qua 21 kernel Gaussian
       exp(-(sim-μ)²/2σ²) → log-pooling → proj_select
   Mask NULL → softmax over N → select_prob (B, N)
   ⇒ "triple nào quan trọng?"
   │
   ▼
③ SENTENCE-LEVEL GAT  (_self_attention, lặp N anchor)
   Với mỗi anchor i: kernel attention token-level giữa anchor và mọi node
   → denoise → trọng số GAT (proj_gat) → mask NULL → softmax
   ⇒ biểu diễn đã khử nhiễu outputs_de (B, N, 768)
   (dùng full-text mask cho cả 2 nhánh — KHỚP KernelGAT gốc)
   │
   ▼
④ PER-NODE CLASSIFICATION
   feat = [pooled, denoised] → proj_inference_de → softmax
   ⇒ per_node_prob (B, N, 2): mỗi node "bỏ phiếu" True/False
   │
   ▼
⑤ AGGREGATE
   agg = Σ_N (select_prob × per_node_prob)
   logits = log(agg)            ← log-probabilities
   │
   ▼
output: {logits (B,2), per_node_pred, node_probs}
```

**Khác biệt duy nhất so với KernelGAT gốc** (chỉ là cơ học, không đổi toán):
1. Node = `[CLS] claim [SEP] triple [SEP]` thay vì câu Wikipedia
2. `mu/sigma` đăng ký bằng buffer (chạy được CPU; bản gốc hard-code `.cuda()`)
3. Mask NULL node ở mọi softmax (cho phép số triple thay đổi)
4. `num_labels=2` (True/False) thay vì 3 (FEVER S/R/NEI)

### 5.3. Baseline công bằng — BertConcatBaseline

Để chứng minh lợi ích đến từ **kiến trúc** chứ không chỉ từ supervised
fine-tuning, thêm baseline cùng triple pool nhưng **bỏ kernel/GAT**:

```
Encode N node → mean-pool [CLS] (bỏ NULL) → Linear → log-softmax → True/False
```

Chọn qua `model_type: kernel | concat_baseline` trong config.

### 5.4. Loss

**File:** [stage3/losses.py](stage3/losses.py)

```python
loss = NLLLoss(logits, label)    # logits là log-prob → NLL khớp
```

Thuần verification loss, **không** có auxiliary loss (đã bỏ để đảm bảo fairness).

---

## 6. Luồng chạy đầu-cuối cho 1 claim

```
Claim: "Obama's wife was born in Chicago"   (gt: True)
   │
   ├─[A · Stage 1] GPT divide
   │     → {"Obama spouse wife":[Obama,Michelle], "wife birthPlace Chicago":[Michelle,Chicago]}
   │
   ├─[A · Stage 2.1] relation_candidates (Algorithm 1)
   │     → sub1: [spouse, child, ...]   sub2: [birthPlace, deathPlace, ...]
   │
   ├─[A · Stage 2.2] GPT top-K
   │     → sub1: [spouse]   sub2: [birthPlace]
   │
   ├─[A · Stage 2.3] build_triple_pool (grounding + bridging)
   │     → [(Obama, spouse, Michelle), (Michelle, birthPlace, Chicago)]
   │        ↑ Michelle được khám phá qua KG (multi-hop)
   │
   ├─[CACHE] cache/stage12_dev.pt
   │  ═════════════════════ GIAI ĐOẠN B ═════════════════════
   │
   ├─[B · data] build_graph → 2 node thật + 8 NULL, pre-tokenize
   │
   ├─[B · ①] BERT encode mỗi (claim, triple)
   │
   ├─[B · ②] Node kernel → select_prob = [0.55, 0.45, 0, 0, ...]
   │
   ├─[B · ③] GAT denoise giữa 2 node
   │
   ├─[B · ④⑤] Per-node vote → aggregate → log P(True)=0.82
   │
   └─► Predict: TRUE ✓
```

---

## 7. Cấu hình (configs/default.yaml)

| Nhóm | Tham số | Giá trị | Ghi chú |
|---|---|---|---|
| Model | `model_type` | kernel | kernel \| concat_baseline |
| | `bert_model` | bert-base-uncased | |
| | `num_kernels` | 21 | giống KernelGAT |
| | `num_labels` | 2 | FactKG True/False |
| Graph | `max_seq_len` | 96 | triple ngắn hơn câu FEVER (130) |
| | `max_nodes` | 10 | KG-GPT trả 5-30 triple |
| | `triple_format` | plain | plain \| separators |
| Train | `batch_size` | 4 | |
| | `gradient_accumulation` | 8 | effective batch = 32 |
| | `learning_rate` | 5e-5 | giống KernelGAT |
| | `num_epochs` | 5 | |
| | `early_stopping_patience` | 3 | |
| Stage 1+2 | `llm_model` | gpt-3.5-turbo-0613 | |
| | `top_k` | 5 | |
| | `max_triples` | 30 | cap triple pool |

---

## 8. Thứ tự thực thi

```bash
# Setup
pip install -r requirements.txt
echo "sk-..." > openai_api_key.txt
# data/: factkg_{train,dev,test}.pickle, dbpedia_2015_undirected_light.pickle,
#        type_dict.pickle (build bằng kg-gpt/data/make_type_dict.py)

# GIAI ĐOẠN A — cache Stage 1+2 (gọi GPT)
python scripts/01_run_stage12.py --split dev --limit 50   # smoke test trước
python scripts/01_run_stage12.py --split dev
python scripts/01_run_stage12.py --split test
python scripts/01_run_stage12.py --split train            # lâu + tốn nhất

# GIAI ĐOẠN B — train + eval
python scripts/02_train_stage3.py --config configs/default.yaml
python scripts/03_evaluate.py --model_path outputs/exp1/best.pt

# Ablation
python scripts/04_ablation.py
```

---

## 9. Ablation

| Variant | Override | Đo điều gì |
|---|---|---|
| `default` | — | Setting tham chiếu |
| `concat_baseline` | `model_type=concat_baseline` | **Kiến trúc kernel có hơn baseline không?** |
| `max_nodes_5/15/20` | `max_nodes` | Độ nhạy với số triple |
| `format_separators` | `triple_format=separators` | Format triple ảnh hưởng? |
| `kernels_11` | `num_kernels=11` | Số kernel ảnh hưởng? |

**Bộ so sánh cốt lõi cho paper:**

| Model | Train signal | Vai trò |
|---|---|---|
| KG-GPT (gốc) | zero/few-shot | baseline ngoài |
| BertConcatBaseline | label | baseline supervised công bằng |
| **KernelKG-GPT** | label | **method đề xuất** |

→ `KernelKG-GPT > ConcatBaseline` chứng minh giá trị **kiến trúc**;
`ConcatBaseline > KG-GPT` chứng minh giá trị của **supervision**.

---

## 10. Tình trạng & lưu ý

### Đã verify (smoke test)
- ✅ Builder: 2-entity trực tiếp, 1-entity expand, **multi-hop bridging**, pool rỗng → không crash
- ✅ Builder **deterministic** (chạy 2 lần giống nhau)
- ✅ Model forward/backward: shape đúng, log-prob hợp lệ, **NULL node prob ≈ 0**, all-NULL không crash
- ✅ Baseline forward hợp lệ
- ✅ 16 file Python pass syntax

### Cần kiểm tra khi chạy thật
- **Schema FactKG:** code đọc `Entity_set`, `Label`, `types` — xác nhận đúng tên key trong pickle
- **`type_dict.pickle`:** phải build trước bằng `kg-gpt/data/make_type_dict.py` (cần file relations.pickle)
- **Coverage:** chạy `--limit 100` để đo % pool rỗng trước khi cam kết chi phí full train
- **Chi phí GPT:** train ~86k claim × 2-3 call/claim — đo thực tế từ smoke test rồi extrapolate
- **Memory:** `batch_size=4 × max_nodes=10` = 40 BERT forward/batch; giảm `max_nodes` nếu OOM

### Giới hạn đã biết
- Chỉ verify được claim có bằng chứng trong DBpedia (KG coverage)
- Pool rỗng → graph toàn NULL → prediction về prior (model học đoán majority)
- `pooler_output` (có tanh) thay vì CLS thuần — lệch nhẹ so với KernelGAT gốc

---

## 11. Tóm tắt đóng góp

1. **Kết hợp 2 framework**: truy hồi LLM (KG-GPT) + suy luận kernel (KernelGAT)
2. **Thay rule-based verify bằng neural reasoning** học được — giải quyết điểm
   yếu của KG-GPT
3. **So sánh công bằng**: giữ nguyên Stage 1+2, chỉ đổi bước verify; có baseline
   supervised để tách bạch đóng góp kiến trúc
4. **Không thêm tín hiệu ngoài 2 paper** — kết quả diễn giải được rõ ràng
