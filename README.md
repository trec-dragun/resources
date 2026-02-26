# TREC 2025 DRAGUN Track: Reusable Resources

Reusable artifacts for the [TREC 2025 DRAGUN (Detection, Retrieval, and Augmented Generation for Understanding News) Track](https://trec-dragun.github.io/).

This repository accompanies:

> Dake Zhang, Mark D. Smucker, and Charles L. A. Clarke. **Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment.** Submitted to SIGIR 2026 Resources Track.

## Overview

The DRAGUN benchmark evaluates assistive RAG systems for news trustworthiness assessment with two tasks:

- **Task 1 - Question Generation:** output 10 ranked investigative questions for a target news article.
- **Task 2 - Report Generation (main task):** output a 250-word report grounded in the [MS MARCO V2.1 Segmented Corpus](https://trec-rag.github.io/annoucements/2025-rag25-corpus/#-ms-marco-v21-segmented-corpus).

Evaluation is **rubric**-based. TREC assessors authored importance-weighted rubrics (questions + expected short answers) for 30 topics, and used them to assess submitted runs. These rubrics represent the information that assessors believe is important for readers to assess an article's trustworthiness.


The DRAGUN package:

1. **Topic set and assessor rubrics** — 30 news articles and their importance-weighted rubrics.
2. **Human judgments** — Assessor judgments for both question generation and report generation.
3. **Participant submissions** — All participating teams' submitted runs.
4. **Baseline system** — An iterative multi-agent baseline RAG system covering both tasks ([2025-starter-kit](https://github.com/trec-dragun/2025-starter-kit)).
5. **LLM-based AutoJudge** (`auto_judge/`) — A few-shot prompting-based LLM judge (using `gpt-oss-120b`) that can score new runs beyond those assessed during the original track.
6. **Scoring script** (`utils/score.py`) — Python script that computes scores from either the human judgments or the AutoJudge assessments.
7. **Assessment guidelines** given to TREC assessors (`TREC_2025_DRAGUN_Track_Assessment_Guidelines.pdf`).


## Repository Structure

```text
.
├── README.md
├── TREC_2025_DRAGUN_Track_Assessment_Guidelines.pdf
├── auto_judge/
│   ├── auto_judge.py
│   ├── system_prompts/
│   │   ├── question_judge.txt
│   │   ├── report_judge.txt
│   │   └── compound_question_check.txt
│   └── output/
├── utils/
│   ├── score.py
│   └── results/
└── data/   # expected layout (not fully available in this repository)
    ├── trec-2025-dragun-topics.jsonl
    ├── human_rubrics/
    ├── human_assessments/
    ├── runs/
    │   ├── question_generation_runs/
    │   └── report_generation_runs/
    └── official_evaluation_results/
```

## Data Availability and Access

Some resources are conventionally hosted on other sites, such as topics, rubrics, human assessments, and raw run files.

| Resource | How to obtain / notes | Expected path |
|---|---|---|
| Topics (30 news articles) | [Download Link](https://trec-dragun.github.io/trec-2025-dragun-topics.jsonl) | `data/trec-2025-dragun-topics.jsonl` |
| Human rubrics (30) | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/human_rubrics/` |
| Human assessments | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/human_assessments/` |
| Raw submitted runs | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/runs/question_generation_runs/`, `data/runs/report_generation_runs/` |
| Official evaluation results | [NIST TREC Browser](https://pages.nist.gov/trec-browser/) | `data/official_evaluation_results/` |
| Participation guidelines | [DRAGUN track website](https://trec-dragun.github.io/) | N/A |
| Organizer baseline system | [trec-dragun/2025-starter-kit](https://github.com/trec-dragun/2025-starter-kit) | N/A |

**Note**: The NIST TREC Browser link for DRAGUN is not live yet. When available, navigate to TREC-34 (2025) and then the DRAGUN section to download the assessor rubrics and judgments, submitted runs, and official evaluation results. The format is similar to the previous [TREC 2024 Lateral Reading Track](https://pages.nist.gov/trec-browser/trec33/lateral/overview/). Access to the raw submitted runs requires an additional step: https://trec.nist.gov/results.html.

## Collection Stats

- Topics: **30**
- Rubric questions: **236** (avg 7.9/topic)
- Rubric answers: **551** (avg 18.4/topic)
- Task 1 submitted runs: **37**
- Task 2 submitted runs: **28**
- Human-assessed Task 1 question pairs: **12,733**
- Human-assessed Task 2 answer-report pairs: **15,428**

## Setup

### Prerequisites

- Python **3.12+**
- GPU(s) and serving stack capable of `gpt-oss-120b` (for AutoJudge generation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Serve the Judge Model (vLLM Example)

If you want to run the judge locally, install vLLM:

```bash
pip install vllm
```

```bash
vllm serve gpt-oss-120b \
    --tensor-parallel-size <number_of_gpus> \
    --port 8000
```

### Configure AutoJudge Endpoint

Edit the configuration constants in `auto_judge/auto_judge.py`:

- `BASE_URL` (e.g., `http://localhost:8000/v1`)
- `MODEL` (default: `openai/gpt-oss-120b`)

## Input File Formats

### Task 1 Run Format (`data/runs/question_generation_runs/<run_tag>`)

- One TSV file per run, **no header**.
- Each row has 5 fields:

```text
topic_id<TAB>team<TAB>run_tag<TAB>run_question_rank<TAB>run_question_text
```

Example:

```text
msmarco_v2.1_doc_04_420132660	dragun-organizers	dragun-organizers-starter-kit-task-1	1	What is Atlas Obscura and Gastro Obscura's reputation among media critics and fact-checkers for accuracy and reliability?
```

### Task 2 Run Format (`data/runs/report_generation_runs/<run_tag>`)

- One JSONL file per run.
- Each line contains a JSON object in the following format:

```json
{
	"metadata": {
		"run_id": "organizers-run-example", 
		"topic_id": "msmarco_v2.1_doc_xx_xxxxx0"
	},
	"responses": [
		{
			"text": "This is the first sentence.",
				"citations": [
					"msmarco_v2.1_doc_xx_xxxxxx1#x_xxxxxx3",
					"msmarco_v2.1_doc_xx_xxxxxx2#x_xxxxxx4"
				]
			},
		{
			"text": "This is the second sentence.",
			"citations": []
		}
	]
}
```

## Example Usage

Run commands from the repository root.

### 1. Generate AutoJudge Assessments

`auto_judge/auto_judge.py` uses organizer baseline runs as few-shot examples (`organizer-gpt-oss-t1`, `dragun-organizers-starter-kit-task-1`, `organizer-t1-perplex`, `organizer-t1-chatgpt`, `organizer-gpt-oss-t2`, `dragun-organizers-starter-kit-task-2`). Keep these run files available in the corresponding `data/runs/...` folders when running AutoJudge.

```bash
# Task 1 preprocessing: detect compound questions
python auto_judge/auto_judge.py \
    --task auto_compound_question_check \
    --input_folder_path ./data/runs/question_generation_runs \
    --output_folder_path ./auto_judge/output

# Task 1: assess question similarity
python auto_judge/auto_judge.py \
    --task auto_question_evaluation \
    --input_folder_path ./data/runs/question_generation_runs \
    --output_folder_path ./auto_judge/output

# Task 2: assess report coverage/contradiction
python auto_judge/auto_judge.py \
    --task auto_report_evaluation \
    --input_folder_path ./data/runs/report_generation_runs \
    --output_folder_path ./auto_judge/output
```

AutoJudge outputs:

- `auto_judge/output/auto_compound_question_check.csv`
- `auto_judge/output/auto_question_assessments.csv`
- `auto_judge/output/auto_report_assessments.csv`

### 2. Compute Run Scores

`utils/score.py` expects explicit input CSV file paths.

```bash
# Question generation score using human assessments
python utils/score.py \
    --task question_generation_evaluation \
    --type human \
    --assessment_input ./data/human_assessments/question_assessments.csv \
    --compound_check_input ./auto_judge/output/auto_compound_question_check.csv \
    --output ./utils/results

# Question generation score using auto assessments
python utils/score.py \
    --task question_generation_evaluation \
    --type auto \
    --assessment_input ./auto_judge/output/auto_question_assessments.csv \
    --compound_check_input ./auto_judge/output/auto_compound_question_check.csv \
    --output ./utils/results

# Report generation score using human assessments
python utils/score.py \
    --task report_generation_evaluation \
    --type human \
    --assessment_input ./data/human_assessments/report_assessments.csv \
    --output ./utils/results

# Report generation score using auto assessments
python utils/score.py \
    --task report_generation_evaluation \
    --type auto \
    --assessment_input ./auto_judge/output/auto_report_assessments.csv \
    --output ./utils/results
```

Generated output files are named:

- `human_question_generation_per_topic_results.csv`
- `human_question_generation_per_run_results.csv`
- `auto_question_generation_per_topic_results.csv`
- `auto_question_generation_per_run_results.csv`
- `human_report_generation_per_topic_results.csv`
- `human_report_generation_per_run_results.csv`
- `auto_report_generation_per_topic_results.csv`
- `auto_report_generation_per_run_results.csv`

### 3. Develop and Test New AutoJudge Systems

Use the DRAGUN 2025 submissions and human judgments as a benchmark to develop and evaluate your own automatic judging system.

1. Obtain the topics, rubrics, human assessments, raw runs, and official evaluation results, and put them in the corresponding folder.
2. Implement your own judging system and produce the following output files:
    - `auto_judge/output/auto_compound_question_check.csv`
    - `auto_judge/output/auto_question_assessments.csv`
    - `auto_judge/output/auto_report_assessments.csv`
3. Run the scoring scripts to compute run-level scores from your automatic assessments:
    ```bash
    python utils/score.py \
        --task question_generation_evaluation \
        --type auto \
        --assessment_input ./auto_judge/output/auto_question_assessments.csv \
        --compound_check_input ./auto_judge/output/auto_compound_question_check.csv \
        --output ./utils/results

    python utils/score.py \
        --task report_generation_evaluation \
        --type auto \
        --assessment_input ./auto_judge/output/auto_report_assessments.csv \
        --output ./utils/results
    ```
4. Compare the resulting run ranking against the official evaluation results (from `data/official_evaluation_results/`) to measure how well your AutoJudge preserves the human-derived ranking.

### 4. Evaluate New RAG Systems

Use the DRAGUN benchmark to evaluate your own RAG system for assistive news trustworthiness assessment.

1. Use your system to generate run files for the 30 DRAGUN topics (following the submission format described in the [participation guidelines](https://trec-dragun.github.io/)) and place them in `data/runs/question_generation_runs/` and/or `data/runs/report_generation_runs/`, together with the organizer baseline runs used as few-shot examples.
2. Obtain the topics, rubrics, and human assessments, and put them in the corresponding folder. 
3. Run the AutoJudge to produce automatic assessments for your runs:
   ```bash
   python auto_judge/auto_judge.py \
       --task auto_compound_question_check \
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
4. Run the scoring scripts to compute scores for your runs:
   ```bash
   python utils/score.py \
       --task question_generation_evaluation \
       --type auto \
       --assessment_input ./auto_judge/output/auto_question_assessments.csv \
       --compound_check_input ./auto_judge/output/auto_compound_question_check.csv \
       --output ./utils/results

   python utils/score.py \
       --task report_generation_evaluation \
       --type auto \
       --assessment_input ./auto_judge/output/auto_report_assessments.csv \
       --output ./utils/results
   ```

## Citation

If you use these resources, please cite:

```bibtex
@misc{zhang2026dragun-resources,
  author = {Dake Zhang and Mark D. Smucker and Charles L. A. Clarke},
  title = {Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment},
  year = {2026}
}
```

You may also cite the DRAGUN overview paper:

```bibtex
@inproceedings{zhang2025dragun-overview,
  author = {Dake Zhang and Mark D. Smucker and Charles L. A. Clarke},
  title = {Overview of the TREC 2025 DRAGUN Track: Detection, Retrieval, and Augmented Generation for Understanding News},
  booktitle = {The Thirty-Fourth Text REtrieval Conference Proceedings (TREC 2025)},
  series = {NIST Special Publication},
  publisher = {National Institute of Standards and Technology (NIST)},
  year = {2025}
}
```

## License

This project is released under [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the materials for any purpose, including commercial use, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.
