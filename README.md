# TREC 2025 DRAGUN Track: Reusable Resources

Reusable data, judgments, and evaluation code from the [TREC 2025 DRAGUN (Detection, Retrieval, and Augmented Generation for Understanding News) Track](https://trec-dragun.github.io/).

This repository accompanies the following paper:

> Dake Zhang, Mark D. Smucker, and Charles L. A. Clarke. **Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment.** Submitted to SIGIR '26 Resource Track.

## Overview

The DRAGUN track evaluates assistive RAG systems that support readers' news trustworthiness assessment through two tasks:

- **Task 1 — Question Generation:** Produce a ranked list of 10 investigative questions that a careful reader should ask of a target news article.
- **Task 2 — Report Generation (Main Task):** Produce a 250-word report grounded in the [MS MARCO V2.1 Segmented Corpus](https://trec-rag.github.io/annoucements/2025-rag25-corpus/#-ms-marco-v21-segmented-corpus) that provides readers with essential background and context for evaluating the article's trustworthiness.

Evaluation is **rubric**-based: TREC assessors created importance-weighted rubrics of investigative questions with expected short answers for 30 news articles. These rubrics represent the information that assessors believe is important for readers to assess an article's trustworthiness.

This repository provides:

1. **Topic set and assessor rubrics** — 30 news articles and their importance-weighted rubrics.
2. **Human judgments** — Assessor judgments for both question generation and report generation.
3. **Participant submissions** — All participating teams' submitted runs.
4. **Baseline system** — An iterative multi-agent baseline RAG system covering both tasks ([2025-starter-kit](https://github.com/trec-dragun/2025-starter-kit)).
5. **LLM-based AutoJudge** — A few-shot prompting-based LLM judge (using `gpt-oss-120b`) that can score new runs beyond those assessed during the original track (Kendall's τ = 0.678 for Task 1; τ = 0.872 for Task 2).
6. **Scoring scripts** — Python scripts that compute scores from either the human judgments or the AutoJudge assessments.

## Repository Structure

```
.
├── auto_judge/
│   ├── auto_judge.py                  # AutoJudge: LLM-based automatic assessment
│   ├── system_prompts/
│   │   ├── question_judge.txt         # Prompt for question similarity judgment
│   │   ├── report_judge.txt           # Prompt for report assessment
│   │   └── compound_question_check.txt # Prompt for compound question detection
│   └── output/                        # AutoJudge output assessments
│       ├── auto_question_assessments.csv
│       ├── auto_report_assessments.csv
│       └── auto_compound_question_check.csv
├── data/
│   ├── trec-2025-dragun-topics.jsonl  # 30 news articles (topics)
│   ├── human_rubrics/                 # 30 assessor-authored rubrics (one per topic)
│   │   └── msmarco_v2.1_doc_*.json    # e.g., msmarco_v2.1_doc_04_420132660.json
│   ├── human_assessments/             # Human judgments from TREC assessors
│   │   ├── question_assessments.csv
│   │   ├── report_assessments.csv
│   │   └── compound_question_check.csv
│   ├── runs/                          # Submitted runs (obtain from NIST; see below)
│   │   ├── question_generation_runs/
│   │   └── report_generation_runs/
│   └── official_evaluation_results/   # Official evaluation results (obtain from NIST; see below)
├── utils/
│   └── score.py                       # Scoring scripts for both tasks
├── LICENSE
└── README.md
```

## Data Availability

Some resources are provided directly in this repository; others are hosted externally by NIST or on the DRAGUN track website.

### Provided in This Repository

| Resource | Path | Description |
|---|---|---|
| AutoJudge code | `auto_judge/auto_judge.py` | LLM-based automatic assessment system |
| System prompts | `auto_judge/system_prompts/` | Prompts used by the AutoJudge |
| AutoJudge assessments | `auto_judge/output/` | LLM-based assessments of existing runs |
| Human rubrics | `data/human_rubrics/` | 30 importance-weighted rubrics (one per topic) |
| Human assessments | `data/human_assessments/` | TREC assessor judgments for both tasks |
| Scoring scripts | `utils/score.py` | Compute scores from human or auto assessments |

### Obtain Externally

| Resource | Where to Get It | Path to Place It |
|---|---|---|
| Topics (30 news articles) | [trec-2025-dragun-topics.jsonl](https://trec-dragun.github.io/trec-2025-dragun-topics.jsonl) | `data/trec-2025-dragun-topics.jsonl` |
| Submitted runs | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/runs/question_generation_runs/` and `data/runs/report_generation_runs/` |
| Official evaluation results | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/official_evaluation_results/` |
| Baseline system | [2025-starter-kit](https://github.com/trec-dragun/2025-starter-kit) | — |

**Note:** The NIST TREC Browser link may not be live yet. When available, navigate to the TREC 2025 DRAGUN Track section to download the submitted runs and official evaluation results. The format is similar to other TREC tracks (e.g., the [TREC 2024 Lateral Reading Track](https://pages.nist.gov/trec-browser/)).

## Use Cases

### Use Case 1: Develop and Test New AutoJudge Systems

Use the DRAGUN 2025 submissions and human judgments as a benchmark to develop and evaluate your own automatic judging system.

1. Obtain the DRAGUN 2025 raw run files from the [NIST TREC Browser](https://pages.nist.gov/trec-browser/) and place them in `data/runs/question_generation_runs/` and `data/runs/report_generation_runs/`.
2. Implement your own judging system and produce the following output files:
   - `auto_judge/output/auto_question_assessments.csv`
   - `auto_judge/output/auto_report_assessments.csv`
   - `auto_judge/output/auto_compound_question_check.csv`
3. Run the scoring scripts to compute run-level scores from your automatic assessments:
   ```bash
   python utils/score.py \
       --task question_generation_evaluation \
       --type auto \
       --input ./auto_judge/output \
       --output ./auto_judge/output

   python utils/score.py \
       --task report_generation_evaluation \
       --type auto \
       --input ./auto_judge/output \
       --output ./auto_judge/output
   ```
4. Compare the resulting run ranking against the official evaluation results (from `data/official_evaluation_results/`) to measure how well your AutoJudge preserves the human-derived ranking.

### Use Case 2: Evaluate New RAG Systems

Use the DRAGUN benchmark to evaluate your own RAG system for assistive news trustworthiness assessment.

1. Use your system to generate run files for the 30 DRAGUN topics (following the submission format described in the [participation guidelines](https://trec-dragun.github.io/)) and place them in `data/runs/question_generation_runs/` and/or `data/runs/report_generation_runs/`.
2. Run the AutoJudge to produce automatic assessments for your runs:
   ```bash
   python auto_judge/auto_judge.py \
       --task compound_question_check \
       --input_folder_path ./data/runs/question_generation_runs \
       --output_folder_path ./auto_judge/output

   python auto_judge/auto_judge.py \
       --task auto_question_evaluation \
       --input_folder_path ./data/runs/question_generation_runs \
       --output_folder_path ./auto_judge/output

   python auto_judge/auto_judge.py \
       --task auto_report_evaluation \
       --input_folder_path ./data/runs/report_generation_runs \
       --output_folder_path ./auto_judge/output
   ```
3. Run the scoring scripts to compute scores for your runs:
   ```bash
   python utils/score.py \
       --task question_generation_evaluation \
       --type auto \
       --input ./auto_judge/output \
       --output ./auto_judge/output

   python utils/score.py \
       --task report_generation_evaluation \
       --type auto \
       --input ./auto_judge/output \
       --output ./auto_judge/output
   ```

## Setup

### Prerequisites

- Python 3.8+
- A GPU with sufficient VRAM to serve `gpt-oss-120b` (e.g., NVIDIA RTX PRO 6000 or equivalent)

### Install vLLM

```bash
pip install vllm
```

### Download and Serve the Model

Download and start a vLLM-compatible OpenAI API server with `gpt-oss-120b`:

```bash
vllm serve gpt-oss-120b \
    --tensor-parallel-size <number_of_gpus> \
    --port 8000
```

The model will be downloaded automatically from Hugging Face on the first run. Adjust `--tensor-parallel-size` based on your GPU setup.

### Configure the Endpoint

In `auto_judge/auto_judge.py`, replace the endpoint string with your vLLM server address (e.g., `http://localhost:8000/v1`).

## Usage

### 1. Compound Question Check (Task 1 Preprocessing)

Before scoring question generation runs, filter out compound questions:

```bash
python auto_judge/auto_judge.py \
    --task compound_question_check \
    --input_folder_path ./data/runs/question_generation_runs \
    --output_folder_path ./auto_judge/output
```

### 2. Automatic Question Evaluation (Task 1)

Score question generation runs using the AutoJudge:

```bash
python auto_judge/auto_judge.py \
    --task auto_question_evaluation \
    --input_folder_path ./data/runs/question_generation_runs \
    --output_folder_path ./auto_judge/output
```

### 3. Automatic Report Evaluation (Task 2)

Score report generation runs using the AutoJudge:

```bash
python auto_judge/auto_judge.py \
    --task auto_report_evaluation \
    --input_folder_path ./data/runs/report_generation_runs \
    --output_folder_path ./auto_judge/output
```

### 4. Compute Scores

Compute final scores from either human or automatic assessments:

```bash
# Score question generation (using human assessments)
python utils/score.py \
    --task question_generation_evaluation \
    --type human \
    --input ./data/human_assessments \
    --output ./data/official_evaluation_results

# Score question generation (using auto assessments)
python utils/score.py \
    --task question_generation_evaluation \
    --type auto \
    --input ./auto_judge/output \
    --output ./auto_judge/output

# Score report generation (using human assessments)
python utils/score.py \
    --task report_generation_evaluation \
    --type human \
    --input ./data/human_assessments \
    --output ./data/official_evaluation_results

# Score report generation (using auto assessments)
python utils/score.py \
    --task report_generation_evaluation \
    --type auto \
    --input ./auto_judge/output \
    --output ./auto_judge/output
```

## Citation

If you use these resources, please cite:

```bibtex
@article{zhang2026dragun-resources,
    author = {Zhang, Dake and Smucker, Mark D. and Clarke, Charles L. A.},
    title = {Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment}
```

You may also wish to cite the TREC 2025 DRAGUN Track overview paper:

```bibtex
@inproceedings{zhang2025dragun-overview,
    author = {Dake Zhang and Mark D. Smucker and Charles L. A. Clarke},
    title = {{Overview of the TREC 2025 DRAGUN Track: Detection, Retrieval, and Augmented Generation for Understanding News}},
    booktitle = {{The Thirty-Fourth Text REtrieval Conference Proceedings (TREC 2025)}},
    series = {{NIST Special Publication}},
    publisher = {{National Institute of Standards and Technology (NIST)}},
    year = {2025}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the materials for any purpose, including commercial use, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.
