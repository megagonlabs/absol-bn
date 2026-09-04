# ABSOL

Code for **ABSOL: Aggregated Bayesian Subsampling Orchestrated with
LLMs**. ABSOL learns Bayesian-network structure by aggregating many GES searches
over row-and-column subsamples, with four optional LLM-guided interventions:
column grouping, adaptive sampling, parent ordering, and cycle arbitration.


## Setup

Create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate absol_bn
```


## Prepare the benchmark data

The paper uses five discrete Bayesian-network benchmarks. Insurance, Hepar2,
Diabetes, and Munin come from the
[bnlearn Bayesian Network Repository](https://www.bnlearn.com/bnrepository/).
Neuropathic comes from the
[Neuropathic Pain Diagnosis Simulator](https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator).

After downloading the BIF files into data/dataset_name/model.bif from the original source,
generate the CSV data like so:

```bash
python scripts/generate_synthetic_data.py data/munin/model.bif --n-samples 1000000
```

Each dataset then has `synthetic/train.csv` and `synthetic/test.csv`. Prediction
question files are optional: structure-only evaluation needs only the test CSV
and ground-truth BIF. To evaluate fitted CPTs as classifiers, generate questions
with `scripts/generate_ground_truth_predictions.py`.


## Run an experiment

Start with the small, non-LLM Insurance example:

```bash
python scripts/train_all_datasets.py configs/example_insurance.json
```

LLM-backed models read the key associated with `llm_provider`:

```bash
export OPENAI_API_KEY=...      # openai (default)
export FIREWORKS_API_KEY=...   # fireworks
export GEMINI_API_KEY=...      # gemini via its OpenAI-compatible endpoint
```

LLM responses are cached in SQLite within the run directory by 
default for efficiency. Changing a prompt, model, or sampling 
parameter changes the cache key.

Outputs are written below `logs/<timestamp>_<config-name>/`. Each model directory
contains the serialized model, resolved config, training log, and evaluation
JSON. Multi-trial runs also produce an aggregated variance report.

```text
logs/<timestamp>_<config-name>/
└── <dataset>/[trial_N/]
    ├── candidate_edges.pkl
    ├── llm_cache.sqlite
    ├── bagging/
    └── llm_bagging_all_augs/
        ├── *.pkl
        ├── config.json
        ├── training.log
        └── evaluation.json
```

Models are structure-only by default (`fit_parameters: false`). Evaluation then
reports BDeu and, when a ground-truth BIF is configured, structural metrics. Set
`fit_parameters: true` to fit conditional probability tables and enable
prediction and classification evaluation.

## Configuration

Top-level config sections are:

- `defaults`: settings shared by datasets and sweep runs;
- `datasets`: input paths and per-dataset overrides; and
- `runs` (optional): named overrides for hyperparameter sweeps.

Merge order is `defaults < dataset < run`. `num_trials > 1` writes `trial_N/`
subdirectories. See the checked-in configs for complete examples.

Canonical model IDs are:

| Model ID | Description |
|---|---|
| `bagging` | Subsample-aggregated GES without LLM guidance |
| `llm_bagging_all_augs` | ABSOL with all four LLM augmentations |
| `llm_bagging_<aug>` | Only one named LLM augmentation |
| `llm_bagging_loo_<aug>` | All augmentations except the named one |
| `ges` | One full-data GES search |
| `pc` | pgmpy PC baseline |
| `prompt_bn` | PromptBN baseline |
| `bfs` | Multi-turn breadth-first baseline |
| `llm_cd` | LLM-assisted conditional-independence baseline |

The three LLM-only baselines are based on the following papers:

- **PromptBN:** Yinghuan Zhang, Yufei Zhang, Parisa Kordjamshidi, and Zijun Cui.
  “Bayesian Network Structure Discovery Using Large Language Models.”
  *Transactions on Machine Learning Research*, 2026.
  [Paper](https://openreview.net/forum?id=G4mrO8LVix).
- **BFSBN** (model ID `bfs`): Thomas Jiralerspong, Xiaoyin Chen, Yash More,
  Vedant Shah, and Yoshua Bengio. “Efficient Causal Graph Discovery Using Large
  Language Models.” *arXiv preprint arXiv:2402.01207*, 2024.
  [Paper](https://doi.org/10.48550/arXiv.2402.01207).
- **LLMCD** (model ID `llm_cd`): Huaming Du, Yujia Zheng, Baoyu Jing, Yu Zhao,
  Gang Kou, Guisong Liu, Tao Gu, Weimin Li, and Carl Yang. “Causal Discovery
  through Synergizing Large Language Model and Data-Driven Reasoning.”
  *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and
  Data Mining*, 2025. [Paper](https://doi.org/10.1145/3711896.3736874).

The augmentation names are `parent_ordering`, `column_grouping`,
`cycle_arbitration`, and `adaptive_bagging`. The model registry and generated
ablation IDs live in `MODEL_DEFS` and `LLM_AUGS` in
`scripts/train_all_datasets.py`.

Graph-refinement variants discussed as negative results in the appendix remain
available as `llm_bagging_all_augs_plus_vanilla_refinement`,
`llm_bagging_all_augs_plus_refinement_w_support`, and
`llm_bagging_all_augs_plus_narrow_refinement`. Graph refinement is deliberately
not part of `all_augs`.

## Interface

Import models through `core.models`:

```python
from core.models import BaggingBayesianNetwork, BaggingConfig

config = BaggingConfig(
    num_samples=100,
    num_rows_per_sample=10_000,
    num_columns_per_sample=15,
    max_parents_per_node=5,
    fit_parameters=False,
)
model = BaggingBayesianNetwork(config).fit(train_df)
model.save("bagging.pkl")
```

All models share a frozen typed config, `fit`, and `save`/`load` lifecycle.
Parameter fitting and inference are in `core/models/inference.py`. Bagging and
LLM Bagging use a Markov-blanket backend that also supports propagation-aware
prediction. The other learners use the global pgmpy backend.

## License

See `LICENSE` for the repository's license and attribution requirements.


## Disclosures:

This software may include, incorporate, or access open source software (OSS) components,
datasets and other third party components, including those identified below. The license terms
respectively governing the datasets and third-party components continue to govern those
portions, and you agree to those license terms may limit any distribution, use, and copying.
You may use any OSS components under the terms of their respective licenses, which may
include BSD 3, Apache 2.0, and other licenses. In the event of conflicts between Megagon Labs,
Inc. (“Megagon”) license conditions and the OSS license conditions, the applicable OSS
conditions governing the corresponding OSS components shall prevail.
You agree not to, and are not permitted to, distribute actual datasets used with the OSS
components listed below. You agree and are limited to distribute only links to datasets from
known sources by listing them in the datasets overview table below. You agree that any right to
modify datasets originating from parties other than Megagon are governed by the respective
third party’s license conditions.
You agree that Megagon grants no license as to any of its intellectual property and patent rights.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS (INCLUDING
MEGAGON) “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. You agree to cease using,
incorporating, and distributing any part of the provided materials if you do not agree with the
terms or the lack of any warranty herein.
While Megagon makes commercially reasonable efforts to ensure that citations in this
document are complete and accurate, errors may occur. If you see any error or omission, please
help us improve this document by sending information to contact_oss@megagon.ai.


### Open Source Software (OSS) Components 

All OSS components used within the product are listed below (including their copyright holders and the license information).

For OSS components having different portions released under different licenses, please refer to the included Upstream link(s) specified for each of the respective OSS components for identifications of code files released under the identified licenses.


| ID | OSS Component Name | Modified | Copyright Holder | Upstream Link | License |
|---:|---|:---:|---|---|---|
| 1 | causal-learn | No | cmu-phil | [link](https://github.com/py-why/causal-learn) | MIT License |
| 2 | networkx | No | NetworkX Developers | [link](https://github.com/networkx/networkx) | BSD License |
| 3 | numpy | No | NumPy Developers | [link](https://numpy.org/) | BSD License | 
| 4 | openai | No | OpenAI | [link](https://github.com/openai/openai-python) | Apache Software License | 
| 5 | pandas | No | AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team | [link](https://pandas.pydata.org/) | BSD License | 
| 6 | pgmpy | No | pgmpy | [link](https://github.com/pgmpy/pgmpy) | MIT License |
| 7 | torch | No | Facebook, Inc  | [link](https://pytorch.org/) | BSD License | 
| 8 | scikit-learn | No | The scikit-learn developers | [link](https://github.com/scikit-learn/scikit-learn) | BSD License|
| 9 | scipy | No | Enthought, Inc. and SciPy Developers | [link](https://github.com/scipy/scipy) | BSD License |
