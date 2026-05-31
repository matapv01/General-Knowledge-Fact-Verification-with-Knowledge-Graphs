# Phan tich du an FactGenius

## 1. Tom tat nhanh

Du an nay la mot he thong fact-checking cho cac claim dang cau van ban, tap trung vao viec xac minh mot phat bieu la `True` hay `False` bang cach ket hop:

- Knowledge Graph, cu the la DBpedia.
- LLM zero-shot, cu the la `Meta-Llama-3-8B-Instruct` chay qua vLLM.
- Fuzzy relation mining de bien cac quan he do LLM goi y thanh quan he hop le trong Knowledge Graph.
- Fine-tuning cac pretrained encoder nhu BERT/RoBERTa cho bai toan binary classification.

Noi ngan gon: du an khong chi dua claim vao model de phan loai, ma co mot pipeline sinh evidence tu Knowledge Graph truoc, sau do dua `Claim + Evidence` vao LLM hoac model BERT/RoBERTa de fact-check.

Ten README day du cua du an la:

`FactGenius: Combining Zero-Shot Prompting and Fuzzy Relation Mining to Improve Fact Verification with Knowledge Graphs`

Day co ve la bai nop cho mon/lop IN5550/IN9550, fact-checking track.

## 2. Cau truc thu muc va file

Repo hien co cac file/thuc muc chinh:

| Duong dan | Vai tro |
| --- | --- |
| `README.md` | Huong dan chay pipeline, mo ta cac baseline va experiment. |
| `requirements.txt` | Danh sach dependency Python chinh. |
| `kg.py` | Lop wrapper cho Knowledge Graph, co ham tim duong di theo relation. |
| `llm_filter_relation.py` | Dung LLM de loc cac relation co kha nang lien quan den claim. |
| `mine_llm_filtered_relation.py` | Fuzzy-match relation LLM sinh ra voi relation that trong DBpedia, sau do tao evidence text. |
| `llm_fact_check.py` | Dung LLM de fact-check claim, co the claim-only hoac claim + evidence. |
| `fine_tune_hf.py` | Fine-tune/evaluate BERT/RoBERTa bang Hugging Face Trainer. |
| `fuzzy_filter_relation_all_one_hop.py` | Script thu nghiem/bo sung de lay one-hop relation tu KG, co tuy chon fuzzy-match voi claim. |
| `llm_v1/` | Dataset da xu ly theo pipeline two-stage fuzzy relation mining. |
| `llm_v1_singleStage/` | Dataset da xu ly theo pipeline single-stage fuzzy relation mining. |
| `llm_v1_jsons.zip` | Ket qua JSON do LLM loc relation, duoc nen lai. |

## 3. Du lieu trong repo

Repo khong chua dataset goc `full/train.csv`, `full/val.csv`, `full/test.csv` hay file DBpedia pickle. Thay vao do, repo da commit cac dataset sau xu ly:

### `llm_v1`

Day la du lieu `Claim + Evidence` tao bang quy trinh two-stage fuzzy relation mining.

| Split | So dong | True | False |
| --- | ---: | ---: | ---: |
| `train.csv` | 86,367 | 42,723 | 43,644 |
| `val.csv` | 13,266 | 6,426 | 6,840 |
| `test.csv` | 9,041 | 4,398 | 4,643 |

Moi dong co schema:

```csv
Sentence,Label
```

Trong do `Sentence` khong chi la claim goc, ma da duoc format thanh:

```text
Claim: ... Evidence: ...
```

Vi du:

```text
Claim: I have heard that Mobyland had a successor. Evidence: Mobyland >- successor -> "Aero 2"
```

### `llm_v1_singleStage`

Day la bien the single-stage. So dong va phan bo label giong `llm_v1`:

| Split | So dong | True | False |
| --- | ---: | ---: | ---: |
| `train.csv` | 86,367 | 42,723 | 43,644 |
| `val.csv` | 13,266 | 6,426 | 6,840 |
| `test.csv` | 9,041 | 4,398 | 4,643 |

Khac biet nam o cach validate/fuzzy-match relation. Single-stage bo qua buoc mo rong relation lan 2 trong `mine_llm_filtered_relation.py`.

### `llm_v1_jsons.zip`

Archive nay chua cac JSON do LLM sinh ra trong buoc loc relation:

- `llm_v1_jsons/llm_train/`
- `llm_v1_jsons/llm_val/`
- `llm_v1_jsons/llm_test/`

Tong so entry trong zip la 108,678. Con so nay bao gom ca thu muc. So file JSON gan voi so dong dataset:

- `llm_train`: khoang 86,367 JSON.
- `llm_val`: khoang 13,266 JSON.
- `llm_test`: khoang 9,041 JSON.

## 4. Pipeline tong the

Co the hinh dung pipeline nhu sau:

```text
Dataset goc FactKG
        |
        v
Claim + entity set
        |
        v
Lay cac relation ung vien tu DBpedia quanh entity
        |
        v
LLM loc relation lien quan den claim
        |
        v
Fuzzy-match relation LLM goi y voi relation that trong DBpedia
        |
        v
KG search tao connected/walkable paths
        |
        v
Tao text: "Claim: ... Evidence: ..."
        |
        +-------------------------------+
        |                               |
        v                               v
Fine-tune BERT/RoBERTa          Zero-shot LLM fact-checking
True/False classifier           True/False answer
```

## 5. Phan tich tung script

### 5.1. `kg.py`

File nay dinh nghia class `KG`, wrapper tren mot dictionary Knowledge Graph da load tu pickle.

Knowledge Graph co dang gan nhu:

```python
kg[head_entity][relation] = [tail_entity_1, tail_entity_2, ...]
```

Class co cac ham chinh:

- `search(ents, rels)`: tim cac duong di trong KG dua tren danh sach entity va relation.
- `walk(start, path, ends=None)`: di tu mot entity theo chuoi relation.
- `get_tail(h, r)`: lay cac tail entity tu head `h` qua relation `r`.

Ket qua `search` gom hai loai:

- `connected`: duong di ket noi duoc entity dau voi mot entity khac trong claim.
- `walkable`: duong di co the di theo relation nhung khong nhat thiet cham den entity dich mong muon.

Mot diem dang chu y: `walk` co dung `choice(...)` khi relation cuoi co nhieu tail, nen mot phan evidence co the khong hoan toan deterministic neu co nhieu ket qua hop le.

### 5.2. `llm_filter_relation.py`

Day la buoc dung LLM de loc relation ung vien.

Input can co:

- Dataset goc, mac dinh `data_path=/fp/projects01/ec30/factkg/full/`.
- DBpedia pickle, mac dinh `dbpedia_path=/fp/projects01/ec30/factkg/dbpedia/dbpedia_2015_undirected_light.pickle`.
- vLLM server expose OpenAI-compatible API.

Voi moi row:

1. Lay `Entity_set` tu dataset.
2. Voi moi entity, lay tat ca relation co trong KG.
3. Goi LLM bang prompt yeu cau chon cac relation lien quan de fact-check claim.
4. LLM phai tra ve Python dict hop le, vi du:

```python
{
  "Dawn_Butler": ["successor"],
  "Paul_Boateng": ["successor"]
}
```

Output:

```text
llm_train/{index}.json
llm_val/{index}.json
llm_test/{index}.json
```

Script co retry toi da 10 lan neu output rong, sai format, hoac dung key kieu `Entity-...`.

### 5.3. `mine_llm_filtered_relation.py`

Day la phan quan trong nhat cua du an: bien relation do LLM loc thanh evidence text co the dua vao classifier.

Quy trinh:

1. Doc dataset goc.
2. Doc JSON output cua `llm_filter_relation.py`.
3. Lay entity that trong `Entity_set`.
4. Fuzzy-match entity LLM tra ve voi entity that.
5. Fuzzy-match relation LLM tra ve voi relation that trong KG bang `thefuzz`.
6. Goi `kg.search(...)` de tim duong di.
7. Format evidence thanh text:

```text
Entity_A >- relation -> Entity_B
```

Neu relation bat dau bang `~`, script dao chieu relation khi format.

Output la CSV:

```csv
Sentence,Label
Claim: ... Evidence: ...,True
```

#### Two-stage vs single-stage

Trong `validateRelation(...)`, neu khong bat `--skip_second_stage`, script co buoc mo rong lan 2:

1. Stage 1: fuzzy-match relation LLM goi y voi relation cua tung entity.
2. Stage 2: lay tap relation da match duoc, tiep tuc scan relation cua cac entity khac de tim relation tuong tu.

Chay two-stage:

```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
```

Chay single-stage:

```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
```

### 5.4. `fine_tune_hf.py`

Script nay fine-tune Hugging Face sequence classification model.

Mac dinh:

- Model: `roberta-base`.
- Epochs: 15.
- Batch size: 32.
- Learning rate: `5e-5`.
- Max length tokenizer: 512.
- Output dir: `./results/{model_name}`.
- Logging qua Weights & Biases project `FactKG_IN9550`.

Dataset duoc doc tu:

```text
{data_path}/train.csv
{data_path}/val.csv
{data_path}/test.csv
```

Script convert dataframe thanh Hugging Face `DatasetDict`, tokenize text trong cot `Sentence`, va train binary classifier.

Metric gom:

- accuracy
- precision
- recall
- f1
- confusion matrix

Sau training, script evaluate tren test set va in classification report.

No cung co logic bao cao theo cac nhom reasoning:

- `negation`
- `num1`
- `multi claim`
- `existence`
- `multi hop`

Tuy nhien phan nay chi dung duoc voi dataset goc co cot metadata. Voi dataset da xu ly trong `llm_v1/` chi co `Sentence,Label`, phan code doc `dfx.Metatada` co nguy co loi. Ngoai ra ten cot co ve bi typo: `Metatada` thay vi `Metadata`.

### 5.5. `llm_fact_check.py`

Script nay dung LLM lam fact checker zero-shot.

Co hai che do:

#### Claim-only

Bat flag:

```bash
--llm_knowledge
```

LLM chi nhan claim, khong nhan evidence tu KG. Prompt noi LLM la fact-checker dua tren kien thuc Wikipedia.

#### Claim + evidence

Khong bat `--llm_knowledge`, script se:

1. Doc JSON evidence do `llm_filter_relation.py` tao.
2. Fuzzy-match entity/relation.
3. Tim duong di trong KG.
4. Dua evidence vao prompt.
5. Yeu cau LLM tra loi True/False va mot cau giai thich.

Output:

```text
llm_prompt_check_{set}.csv
llm_prompt_check_{set}_llm_knowledge.csv
```

### 5.6. `fuzzy_filter_relation_all_one_hop.py`

Day co ve la script phu/thu nghiem de lay relation one-hop tu KG ma khong can LLM loc relation.

No co hai che do:

- Mac dinh: lay tat ca connected/walkable one-hop paths quanh entity.
- `--claim_match`: token hoa claim, bo stopword, fuzzy-match token voi relation trong KG.

Script nay co mot so dau hieu chua hoan thien:

- Import `nltk` nhung `requirements.txt` khong co `nltk`.
- Can file `tmp_dict.pickle`, nhung repo hien khong co file nay.
- Ten comment cuoi file goi `llm_filter_relation_all_one_hop.py`, trong khi file that la `fuzzy_filter_relation_all_one_hop.py`.

## 6. Cach chay theo README

### 6.1. Cai dependency

```bash
pip install -r requirements.txt
```

Luu y: `requirements.txt` hien thieu it nhat:

```text
thefuzz
nltk
tqdm
```

Trong do:

- `thefuzz` duoc dung trong `mine_llm_filtered_relation.py` va `fuzzy_filter_relation_all_one_hop.py`.
- `nltk` duoc dung trong `fuzzy_filter_relation_all_one_hop.py`.
- `tqdm` duoc import trong `mine_llm_filtered_relation.py`.

Neu chay script mining relation, can cai them:

```bash
pip install thefuzz tqdm
```

Neu chay script one-hop phu, can them:

```bash
pip install nltk
```

### 6.2. Chay vLLM server

README dung:

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct
```

Can GPU manh, README ghi A100 80GB VRAM. Cung can quyen truy cap model tren Hugging Face.

### 6.3. Fine-tune model voi du lieu da xu ly san

Vi repo da co `llm_v1/` va `llm_v1_singleStage/`, co the chay thang:

```bash
python fine_tune_hf.py --model roberta-base --batch_size 32 --data_path ./llm_v1/
```

Hoac:

```bash
python fine_tune_hf.py --model roberta-base --batch_size 32 --data_path ./llm_v1_singleStage/
```

Nhung can chu y rui ro loi o phan evaluate theo metadata nhu da noi o tren.

### 6.4. Tao lai du lieu `llm_v1`

Neu muon tao lai tu dau, can co dataset goc va DBpedia pickle. Sau do:

```bash
python llm_filter_relation.py --set train --vllm_url http://host:8000
python llm_filter_relation.py --set val --vllm_url http://host:8000
python llm_filter_relation.py --set test --vllm_url http://host:8000
```

Giai nen JSON co san:

```bash
unzip llm_v1_jsons.zip
```

Tao CSV two-stage:

```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
python mine_llm_filtered_relation.py --set val --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
python mine_llm_filtered_relation.py --set test --outputPath ./llm_v1/ --jsons_path ./llm_v1_jsons/
```

Tao CSV single-stage:

```bash
python mine_llm_filtered_relation.py --set train --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
python mine_llm_filtered_relation.py --set val --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
python mine_llm_filtered_relation.py --set test --outputPath ./llm_v1_singleStage/ --jsons_path ./llm_v1_jsons/ --skip_second_stage
```

## 7. Ket qua/muc tieu nghien cuu ma repo huong toi

Repo nay so sanh nhieu cach fact-check:

1. LLM zero-shot claim-only.
2. RoBERTa/BERT claim-only baseline.
3. LLM zero-shot voi evidence tu KG.
4. BERT/RoBERTa fine-tuned tren claim + evidence sinh bang single-stage fuzzy relation mining.
5. BERT/RoBERTa fine-tuned tren claim + evidence sinh bang two-stage fuzzy relation mining.

Y tuong nghien cuu chinh:

- LLM co kha nang hieu claim va goi y relation lien quan.
- Knowledge Graph cung cap evidence co cau truc.
- Fuzzy matching giup noi output tu nhien/khong chinh xac cua LLM voi schema relation that trong DBpedia.
- Model encoder nho hon nhu BERT/RoBERTa co the duoc fine-tune tren input da giau evidence de dat ket qua tot hon claim-only.

## 8. Cac diem manh cua du an

- Pipeline co y tuong ro: dung LLM lam bo loc relation, khong dung LLM lam tat ca moi viec.
- Co cache JSON LLM output, giup khong phai goi LLM lai moi lan.
- Co san dataset da xu ly trong repo, nen co the fine-tune/evaluate nhanh hon neu khong co GPU chay Llama.
- Co so sanh single-stage va two-stage relation mining.
- Co ho tro evaluate theo cac nhom reasoning kho nhu negation, multi-hop, multi-claim.

## 9. Rui ro, loi tiem an va diem can luu y

### 9.1. Khong day du de tai lap tu dau

Repo khong chua:

- Dataset goc `full/train.csv`, `full/val.csv`, `full/test.csv`.
- DBpedia pickle.
- `tmp_dict.pickle` cho script one-hop.

Vi vay neu chi clone repo nay, nguoi dung chu yeu chay duoc tren CSV da xu ly san, khong tai tao full pipeline tu raw data duoc.

### 9.2. `requirements.txt` thieu dependency

Can bo sung:

```text
thefuzz
tqdm
nltk
```

Neu khong, cac script mining/fuzzy se loi import.

### 9.3. `fine_tune_hf.py` co kha nang loi voi `llm_v1/`

Sau khi predict test set, script doc lai:

```python
dfx = pd.read_csv(args.data_path + 'test.csv')
dfx['Metadata'] = [ast.literal_eval(e) for e in dfx.Metatada]
```

Nhung `llm_v1/test.csv` chi co:

```csv
Sentence,Label
```

Khong co cot `Metatada` hay `Metadata`. Do do script co the train va predict xong, nhung loi o phan report theo reasoning type.

### 9.4. Ten cot `Metatada` co the la typo

Trong ca `fine_tune_hf.py` va `llm_fact_check.py`, code dung:

```python
dfx.Metatada
```

Ten dung kha nang cao la `Metadata`. Neu dataset goc that su co typo `Metatada` thi khong sao, nhung neu khong thi se loi.

### 9.5. Xu ly path tren Windows co the loi

Nhieu doan tach filename bang:

```python
file.split('/')[-1]
```

Tren Windows, path separator la `\`, nen nen thay bang:

```python
os.path.basename(file)
```

hoac `pathlib.Path(file).stem`.

### 9.6. LLM output parse bang `ast.literal_eval` va regex

`llm_filter_relation.py` lay dict bang:

```python
re.findall(r'\{.*?\}', text, re.DOTALL)[0]
ast.literal_eval(...)
```

Cach nay nhanh nhung de vo neu LLM sinh text co brace long nhau, comment la, hoac format khong dung Python literal. Nen chuyen sang yeu cau JSON nghiem ngat va parse bang `json.loads`.

### 9.7. Ket qua KG search co yeu to ngau nhien

Trong `kg.walk`, khi relation cuoi co nhieu tail:

```python
choice(list(ts.keys()))
```

Dieu nay lam evidence `walkable` co the thay doi giua cac lan chay.

### 9.8. Multiprocessing ghi file song song

Cac script LLM dung multiprocessing de goi API va ghi JSON. Cach nay hop ly de tang toc, nhung can can than voi:

- Rate limit/API server overload.
- File dang ghi do dang neu process bi kill.
- Retry lien tuc voi `wait_fixed=0` co the spam server.

### 9.9. Prompt LLM co vai loi chinh ta/encoding

Trong `llm_fact_check.py`, co chuoi:

```text
Now letâ€™s verify
```

Day la loi encoding cua dau apostrophe. Khong nghiem trong, nhung nen sua thanh:

```text
Now let's verify
```

## 10. De xuat cai thien

Neu muon bien repo nay thanh mot du an de nguoi khac chay lai de dang, nen uu tien:

1. Bo sung `requirements.txt` cho day du: `thefuzz`, `tqdm`, `nltk`.
2. Tao `config.example.yaml` hoac `.env.example` cho cac path nhu DBpedia, data, vLLM URL.
3. Sua path handling bang `pathlib`.
4. Tach code train/evaluate thanh function, tranh execute khi import.
5. Sua `fine_tune_hf.py` de neu dataset khong co metadata thi bo qua report theo reasoning type.
6. Chuan hoa output LLM la JSON that, parse bang `json.loads`.
7. Them seed hoac bo `random.choice` trong `KG.walk` neu can reproducibility.
8. Them README phan "what is included vs not included" de noi ro repo co CSV processed nhung khong co raw FactKG/DBpedia pickle.
9. Them mot script `analyze_data.py` hoac notebook nho de thong ke dataset va preview evidence.
10. Them test don vi nho cho `KG.search`, `paths_to_str2`, `fuzzy_matchEntities`, `validateRelation`.

## 11. Ket luan

FactGenius la mot prototype/nghien cuu ve fact verification bang cach ket hop LLM va Knowledge Graph. Dong gop trung tam cua no nam o viec dung LLM de loc relation co kha nang lien quan, sau do dung fuzzy matching de dua cac relation do ve schema that cua DBpedia, tao evidence text cho classifier.

Neu chi can su dung repo hien tai, huong thuc kha thi nhat la fine-tune/evaluate tren `llm_v1/` hoac `llm_v1_singleStage/` da co san. Neu muon tai lap pipeline tu dau, can bo sung dataset goc, DBpedia pickle, vLLM server, va sua mot so van de dependency/path/metadata nhu da liet ke o tren.
