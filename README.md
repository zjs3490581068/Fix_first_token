# Fix First Token

Code for studying how a fixed initial token affects mathematical reasoning in large language models. The experiments use Qwen3-4B-Base on MATH-500, with a GRPO checkpoint as a comparison.

[中文说明](README.zh-CN.md)

The repository provides label-free cue search, token-ID prefix interventions, repeated sampling, answer grading, pass@k analysis, and sample extraction for blinded process evaluation. Datasets, model weights, generated answers, and private judge mappings are supplied separately.

## Installation

Use Linux, Python 3.11, and a CUDA-compatible NVIDIA GPU for inference. Install PyTorch and vLLM builds appropriate for your CUDA environment; the pinned files describe the tested software stack.

```bash
git clone https://github.com/zjs3490581068/Fix_first_token.git
cd Fix_first_token
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CPU grading, analysis, and development:

```bash
python -m pip install -r requirements-dev.txt
```

`tmux` is optional and must be available on the host if using the background launcher.

## Inputs and Configuration

Download the checkpoints from [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) and [Thinking-Space/Qwen3-4B-Base-GRPO](https://huggingface.co/Thinking-Space/Qwen3-4B-Base-GRPO). Each local model directory must contain the full checkpoint, tokenizer, configuration, and safetensors index.

Prepare MATH-500 and a separate MATH training probe file in the JSONL format used by [cues](https://github.com/sophicle/cues). A schema example, rather than a benchmark record:

```json
{"problem_key":"example_001","problem":"What is 1 + 1?","gt":"2","level":1,"capability":"algebra"}
```

The main experiment requires 500 unique test problems and at least 100 probe problems. Search and generation receive only the problem ID and question; answer labels are used for grading. The preflight rejects search/test question overlap.

The example configuration expects this layout:

```text
Fix_first_token/
  configs/main.json
  data/math500.jsonl
  data/math_train_probe.jsonl
  models/Qwen3-4B-Base/
  models/Qwen3-4B-Base-GRPO/
  outputs/
```

Alternatively, copy `configs/main.json` to `configs/local.json` and edit the input/output paths and GPU indices. `local.json` is ignored by Git. Paths are resolved relative to the configuration file that declares them. Supplementary configurations inherit `main.json`; use the same `extends` mechanism for your own overrides.

The default GPU list is `[0]`. Set, for example, `[0, 1]` for two independent workers. Memory checks and engine resource settings are configurable; tune them for your hardware before a full run. The example retains a 16,384-token output budget and a 17,408-token context window.

## Main Experiment

| Group | Model | Initial output | Samples per problem |
|---|---|---|---:|
| A | Base | Unconstrained | 128 |
| B | Base | Chinese `解答`, ID 106185 | 128 |
| C | Base | ` To`, ID 2014 | 128 |
| D | Base | Cue selected by label-free search | 128 |
| E | GRPO | Unconstrained | 128 |

```bash
python -m first_token run --config configs/main.json --run-dir outputs/main
```

The coordinator performs preflight checks, smoke and long-budget checks, proposal generation, entropy screening, winner freezing, formal sampling, grading, and analysis. When D selects the same token as C, both labels share the same samples. They are not pooled or treated as independent replications.

Individual stages are also available:

```bash
python -m first_token preflight --config configs/main.json --run-dir outputs/main
python -m first_token search --config configs/main.json --run-dir outputs/main
python -m first_token sample --config configs/main.json --run-dir outputs/main
python -m first_token grade --config configs/main.json --run-dir outputs/main
python -m first_token summarize --config configs/main.json --run-dir outputs/main
```

`sample` and `grade` accept `--groups A B C`. Sampling requires a preflight manifest, and D requires the frozen search winner. Reuse a run directory only with the same configuration and code.

### Cue Search

The first 100 probe questions propose candidates. Beam width and expansion size are 20; depths 1 and 2 retain at most 20 proposals each. Proposals must occur on at least half the questions and are ranked by mean original probability over all 100. Depth-2 proposals are projected to their first ID and deduplicated with depth-1 proposals.

Each candidate and the unconstrained baseline are screened on the first 30 probe questions with 16 samples per question. Selection minimizes mean answer-group Shannon entropy. Candidates whose missing-answer rate exceeds the baseline by more than 10 percentage points are rejected; exact entropy ties use token-ID order. Incomplete conditions cannot produce a winner.

## Supplementary Experiments

### Four English Initial Tokens

Each condition generates one answer per problem. The leading spaces below belong to the fixed token.

| Cue | Token ID |
|---|---:|
| ` Solution` | 12478 |
| ` Proof` | 36991 |
| ` Let` | 6771 |
| ` I` | 358 |

```bash
python -m experiments.english_tokens \
  --config configs/english_tokens.json --run-dir outputs/english_tokens
```

### Two-Token Prefix

```bash
python -m experiments.two_token \
  --config configs/two_token.json --run-dir outputs/two_token
```

This fixes ` To determine`, IDs `[2014, 8253]`, and generates one answer per problem. Both supplementary entry points accept optional `--baseline-run outputs/main` for matched comparisons against rollout zero of a completed main experiment. Without this option, they run independently.

If you use a custom supplementary configuration, retain `rollouts=1` and `rollouts_per_batch=1`. Single-sample experiments report pass@1 only.

## Blinded Evaluation Inputs

```bash
python -m experiments.extract_judge_samples \
  --plan sampling_plan.md \
  --run-dir outputs/main \
  --references data/math500_official.jsonl \
  --out outputs/judge_samples
```

The extraction scheme fixes 50 questions with difficulty quotas `4, 9, 11, 13, 13` for levels 1 through 5. Supply a Markdown selection table with columns `ordinal | level | problem_id | sample_index`, where IDs match the run and sample indices are in `[0, 127]`. A/B/C use the same selected index for each question, producing 150 complete responses. The official reference JSONL must contain `problem`, `answer`, `solution`, `level`, `subject`, and `unique_id`, aligned with the test set.

`blind/` and `blind_judge_inputs.zip` contain only anonymous IDs, questions, reference answers/solutions, and complete model responses. `private/` contains group labels, source mappings, automatic grades, and audit information. Extraction preserves prefixes, languages, and long responses. It does not call an AI judge.

## Generation and Evaluation Protocol

No chat template is used. The exact prompt is:

```text
Problem: {problem}
Please reason step by step, and put your final answer within \boxed{}.
Solution:
```

There is no trailing space or newline after `Solution:`. Prefixes are appended as token IDs.

- Temperature `0.6`, top-p `0.95`, no extra top-k restriction, repetition penalty `1`, and neutral vLLM generation defaults.
- Total output budget: 16,384 tokens, including fixed prefix IDs. Continuation limits are 16,384 / 16,383 / 16,382 for free / one-token / two-token conditions.
- Stop IDs: `151643`, `151645`. Stop strings: `\nProblem:`, `\nProblem :`. Both checkpoints use the Base tokenizer after compatibility checks.
- Rollout `r` uses seed `20260819+r`. Batching and resume preserve sample indices; GPU scheduling can still affect exact text reproducibility.
- BF16, one model replica per GPU, eager execution, and at most 16 active sequences. FlashInfer sampling and DeepGEMM are disabled to keep the sampling implementation consistent.
- The last complete balanced `\boxed{}` is extracted, falling back to the last `Final answer`. Math-Verify checks equivalence; extraction, parsing, and verification failures count as incorrect with separate statuses.
- pass@k uses `1 - C(n-c,k)/C(n,k)`, averaged over problems. Confidence intervals use 1,000 problem-level bootstrap resamples with seed 0.

Main analysis exports summary JSON, per-problem CSV, paired comparisons, pass@k curves, first-token frequency plots, and case candidates. Selected illustrative cases are distinct from the fixed blinded sample. Exact GRPO training/test overlap remains unverified.

## Storage and Background Runs

Raw and graded samples are stored separately as compressed JSONL. Atomic shard completion, unique sample keys, and experiment fingerprints support resume. Partial files do not count as completed samples. Worker caches and temporary files are created below the run's `.runtime/` directory.

```bash
bash scripts/launch_tmux.sh main_run configs/main.json
```

The launcher uses Python from the active environment and prints the command, output directory, GPUs, and attach command. Logs, coordinator status, and exit codes are saved with the run. Direct `python -m first_token run ...` commands can resume after a tmux session ends.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use CPU and synthetic fixtures. Tokenizer encoding tests are skipped when the configured model is absent; include them without loading GPU weights by supplying a downloaded tokenizer:

```bash
python -m pytest -q --tokenizer-path=models/Qwen3-4B-Base
```

## Attribution and License

Search, prompts, answer normalization, grading, and metrics are adapted from [sophicle/cues](https://github.com/sophicle/cues), commit `cd9caabe748307df0565502f630e3a834a9f998d`. Sampling and pass@k were cross-checked against [Limit-of-RLVR](https://github.com/LeapLabTHU/Limit-of-RLVR), commit `79c348f4543330bb78b01a5332df09fea2700f70`. See [NOTICE.md](NOTICE.md) for the adaptation details.

Released under the [MIT License](LICENSE). The original cues notice is retained in [LICENSE.cues](LICENSE.cues).
