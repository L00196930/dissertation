# Artifacts : Cloud-Native AI-Based Log Correlation for Distributed Trace Reconstruction

**Namratha Shetty · L00196930**
M.Sc. in Computing in Cloud Technologies · Department of Computing, ATU Donegal
Supervised by Dr. Saim Ghafoor

---

## 1. What is in this submission

Four Jupyter notebooks. Together they implement the identifier-free trace-reconstruction
pipeline described in Chapters 3 and 4 of the dissertation, and produce every result and
figure reported in Chapter 5.

| Notebook | Representation | Dataset |
|---|---|---|
| `Pipeline1_Word2Vec_FINAL.ipynb` | Word2Vec (100-d, trained on the log corpus) | HDFS |
| `Pipeline2_BERT_FINAL.ipynb` | BERT (`bert-base-uncased`, 768-d) | HDFS |
| `Pipeline1_Word2Vec_TraceBench.ipynb` | Word2Vec | TraceBench |
| `Pipeline2_BERT_TraceBench.ipynb` | BERT | TraceBench |

The two pipelines are deliberately identical at every stage except one — the step that turns
a cleaned message into a vector. Holding everything else constant is what allows the
comparison in Chapter 5 to be read as a fair test rather than a confound.

---

## 2. What the pipeline does

Each notebook runs the same sequence:

1. **Retrieve** the benchmark log programmatically.
2. **Capture the ground truth.** The block identifier (HDFS) or task identifier (TraceBench)
   is extracted *before* any text is altered and stored in a separate column.
3. **Clean and mask.** The identifier is removed from the message text, along with volatile
   tokens. An assertion fails loudly if any trace of it survives, so the reconstruction is
   performed blind.
4. **Embed** each cleaned message as a vector (Word2Vec or BERT).
5. **Fuse** that vector with a scaled timestamp: PCA to 50 dimensions, standardise,
   unit-normalise, then append `time_weight × time`.
6. **Cluster** the fused vectors with DBSCAN and assign each cluster a **virtual Trace ID**.
7. **Evaluate** against the withheld identifier — ARI, NMI, homogeneity, completeness,
   a pairwise confusion matrix, and the silhouette score.
8. **Sweep** the neighbourhood radius against the temporal weight.
9. **Ablate** the two signals — meaning only, time only, combined — with the radius retuned
   per configuration so no arrangement is disadvantaged.

---

## 3. Environment

Written for **Google Colab with a GPU runtime**. The BERT notebooks are impractical on a
processor alone; the Word2Vec notebooks run comfortably without one.

Dependencies are installed by the first cell of each notebook:

```
gensim  scikit-learn  pandas  numpy  tqdm  kagglehub      # Word2Vec notebooks
transformers  torch  scikit-learn  pandas  numpy  tqdm    # BERT notebooks
```

---

## 4. Data

Both datasets are downloaded by the notebooks; nothing needs to be placed by hand.

**HDFS** — from the LogHub collection via `kagglehub`:

```
ayenuryrr/loghub-hdfs-hadoop-distributed-file-system-data   (~1.55 GB)
```

The notebooks select `HDFS_v1/HDFS.log` (1.58 GB, 11,175,629 lines) explicitly. This matters:
the archive also contains larger `HDFS_v2` node logs whose block mentions span days under the
same templates and therefore do **not** form clean traces, and a 2,000-line sample that has too
few lines per block to support reconstruction at all.

**TraceBench** — from Zenodo:

```
https://zenodo.org/records/8196385/files/HDFS_v3_TraceBench.zip   (~3 GB)
```

The notebooks concatenate the 364 per-scenario `event.csv` files, giving 11,427,676 events
across 4,502,957 task identifiers.

---

## 5. How to run

1. Open a notebook in Colab and set **Runtime → Change runtime type → GPU**.
2. Run all cells in order. The download step takes a few minutes on first run.
3. Nothing needs editing. Every value that affects the outcome is gathered in the single
   configuration cell near the top:

```python
SEED            = 42
SAMPLE_BLOCKS   = 300     # whole blocks / traces sampled
MIN_BLOCK_LINES = 6       # ignore anything too short to be a trace
PCA_DIM         = 50      # semantic dimensions before fusion
TIME_UNIT       = 60.0    # seconds per unit of the time feature
EPS             = 1.5     # DBSCAN neighbourhood radius
MIN_SAMPLES     = 4       # minimum points for a DBSCAN core
TIME_WEIGHT     = 2.0     # weight of time against meaning
```

Sampling is by **whole block or trace**, never by line, because a trace can only be recovered
if every line composing it is present.

---

## 6. Outputs

Each run writes, into the Colab working directory:

- `reconstructed_traces_word2vec.csv` / `reconstructed_traces_bert.csv` — the full line-by-line
  reconstruction, with columns `message`, `t`, `true_block`, `virtual_trace_id`
- `emb_w2v_<hash>.npy` / `emb_bert_<hash>.npy` — cached embeddings, so repeating the sweep and
  the ablation does not recompute them
- all figures, inline

The two HDFS CSVs are submitted alongside the dissertation and appear as Tables 5.2 and 5.3.

---

## 7. Results these notebooks produce

| Run | ARI | NMI | Pairwise F1 | Clusters | Noise |
|---|---|---|---|---|---|
| Word2Vec · HDFS | 0.2117 | 0.7895 | 0.2152 | 425 | 7.0% |
| BERT · HDFS | 0.2199 | 0.7943 | 0.2235 | 369 | 6.3% |
| Word2Vec · TraceBench | 0.1160 | 0.8011 | 0.1164 | 2,257 | 20.8% |
| BERT · TraceBench | 0.1125 | 0.7976 | 0.1129 | 2,094 | 24.8% |

Ablation on HDFS, radius retuned per configuration:

| Configuration | Word2Vec | BERT |
|---|---|---|
| meaning only | 0.0118 | 0.0020 |
| time only | 0.2076 | 0.2076 |
| combined | 0.2117 | 0.2225 |

---

## 8. Known issues and reproducibility notes

Recorded here so that anyone re-running this work knows what to expect.

1. **Word2Vec is not fully deterministic.** It is trained with `workers=4`, and multi-threaded
   training makes the result vary slightly between runs despite the fixed seed. A re-run
   reproduces BERT exactly but gives a Word2Vec HDFS ARI near 0.2086 rather than 0.2117. Set
   `workers=1` for a strictly reproducible run, at some cost in speed.

2. **The TraceBench sweep and ablation cells were not executed.** In both TraceBench notebooks
   the sweep and ablation cells are present but carry no output. The TraceBench figures reported
   in the dissertation therefore come from the single default configuration only.

3. **The TraceBench runs inherit HDFS parameters.** `TIME_WEIGHT = 2.0` suits the short bursts of
   an HDFS block but pushes a long TraceBench trace past the neighbourhood radius, splitting it.
   The TraceBench figures consequently understate what that dataset can support. A lower temporal
   weight, matched to trace duration, is the correction identified in Section 4.12.

4. **Output filenames collide.** The TraceBench notebooks save under the same names as the HDFS
   notebooks. Rename the output in one of them before running both in the same session, or the
   second will overwrite the first. In the submitted set, the BERT TraceBench save cell also
   raised `NameError: name 'work' is not defined`, so no CSV was written for that run.

5. **One chart is mislabelled.** The sensitivity plot in `Pipeline1_Word2Vec_FINAL.ipynb` carries
   the hard-coded title *"BERT pipeline — ARI sensitivity"*. The data plotted is the Word2Vec
   sweep; only the title string is wrong.

6. **TraceBench columns are misaligned in the source CSVs.** As loaded, `TID` holds the operation
   name, `OpName` holds a timestamp and `EndTime` holds an IP address. The text passed to the
   embedding is therefore partly numeric, which limits what the semantic signal can contribute on
   that dataset.

7. **The TraceBench time scale is not seconds.** The unit-detection branch divides `StartTime` by
   1e6, producing a reported span of 4,966,340,289 "seconds" — roughly 157 years. This inflates
   the temporal feature and contributes to the fragmentation described in Chapter 5.

---

## 9. Mapping to the dissertation

| Dissertation section | Where it lives in the notebooks |
|---|---|
| 3.5 / 4.5 Parsing, masking, leakage prevention | `clean()` and the assertion that follows it |
| 3.6 / 4.6 Semantic representation | the embedding cell — the only stage that differs |
| 3.7 / 4.7 Temporal fusion | `build_features()` |
| 3.8 / 4.8 Clustering and virtual Trace ID | the DBSCAN cell |
| 3.10 / 4.9 Evaluation | `run_pipeline()` |
| 3.12 / 4.10 Parameter sensitivity | the sweep cell |
| 3.11 / 4.11 Ablation | `build_mode()` and `tune()` |
