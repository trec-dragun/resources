import os
import re
import glob
import json
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Literal
from pydantic import BaseModel


# ── Configuration ────────────────────────────────────────────────────────────
BASE_URL = "http://mooneye.cs.uwaterloo.ca:8000/v1"  # Replace it with your vLLM server address
MODEL = "openai/gpt-oss-120b"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
# ─────────────────────────────────────────────────────────────────────────────

client = openai.OpenAI(base_url=BASE_URL, api_key="EMPTY")


def call_llm(system_prompt, user_input, response_schema, schema_name="assessment"):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": response_schema},
        },
        temperature=0,
        top_p=1,
    )
    reasoning = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    # Sanitize control characters that may break JSON parsing
    content = re.sub(r"[\x00-\x1f\x7f]", " ", content)
    return reasoning, content


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CompoundAssessment(BaseModel):
    reasoning: str
    compound_question: bool


class QuestionAssessment(BaseModel):
    rationale: str
    assessment_decision: Literal["very-similar", "similar", "different", "very-different"]


class ReportAnswerAssessment(BaseModel):
    answer_id: str
    reasoning: str
    assessment_decision: Literal["supports", "partial", "contradicts", "none"]


class ReportAssessments(BaseModel):
    assessments: list[ReportAnswerAssessment]


# ── Load shared data ─────────────────────────────────────────────────────────

def load_articles():
    articles = {}
    with open(os.path.join(DATA_DIR, "trec-2025-dragun-topics.jsonl")) as f:
        for line in f:
            data = json.loads(line)
            articles[data["docid"]] = data
    return articles


def load_rubrics():
    rubrics = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "human_rubrics", "*.json"))):
        with open(path) as f:
            data = json.load(f)
        rubrics[data["topic_id"]] = data["rubrics"]
    return rubrics


# ── Task: compound_question_check ────────────────────────────────────────────

def run_compound_question_check(input_folder, output_folder):
    system_prompt = open(os.path.join(SCRIPT_DIR, "system_prompts", "compound_question_check.txt")).read()

    rows = []
    for path in sorted(glob.glob(os.path.join(input_folder, "*"))):
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["topic_id", "team", "run_tag", "rank", "question_text"])
        df["run_tag"] = os.path.basename(path)
        rows.append(df)
    questions = pd.concat(rows, ignore_index=True)

    outputs = []
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="compound_question_check"):
        user_input = (f"Question: {row['question_text']}\n"
                      f"Check if this question is a compound question. Reason first and then decide.")
        reasoning, content = call_llm(system_prompt, user_input, CompoundAssessment.model_json_schema())
        result = CompoundAssessment.model_validate_json(content)
        outputs.append({
            "topic_id": row["topic_id"],
            "run_tag": row["run_tag"],
            "rank": row["rank"],
            "question_text": row["question_text"],
            "compound_question": result.compound_question,
            "reasoning": result.reasoning,
        })

    pd.DataFrame(outputs).to_csv(os.path.join(output_folder, "auto_compound_question_check.csv"), index=False)


# ── Task: auto_question_evaluation ───────────────────────────────────────────

def run_auto_question_evaluation(input_folder, output_folder):
    system_prompt = open(os.path.join(SCRIPT_DIR, "system_prompts", "question_judge.txt")).read()
    articles = load_articles()
    rubrics = load_rubrics()
    human_assessments = pd.read_csv(os.path.join(DATA_DIR, "human_assessments", "question_assessments.csv"))

    # Load participant questions
    rows = []
    for path in sorted(glob.glob(os.path.join(input_folder, "*"))):
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["topic_id", "team", "run_tag", "run_question_rank", "run_question_text"])
        df["run_tag"] = os.path.basename(path)
        rows.append(df)
    all_questions = pd.concat(rows, ignore_index=True)

    # Identify organizer baseline runs (used as few-shot examples, excluded from judging)
    organizer_runs = sorted(set(human_assessments["run_tag"].unique()) & set(all_questions["run_tag"].unique()))

    # Build few-shot examples from organizer baselines + human assessments
    example_questions = all_questions[all_questions["run_tag"].isin(organizer_runs)].copy()
    examples = {}
    for topic_id in sorted(human_assessments["topic_id"].unique()):
        topic_assessments = human_assessments[
            (human_assessments["topic_id"] == topic_id) &
            (human_assessments["run_tag"].isin(organizer_runs))
        ]
        topic_questions = example_questions[example_questions["topic_id"] == topic_id]
        for rq_rank in sorted(topic_assessments["rubric_question_rank"].unique()):
            part = topic_assessments[topic_assessments["rubric_question_rank"] == rq_rank]
            merged = pd.merge(topic_questions, part,
                              on=["topic_id", "run_tag", "run_question_rank", "run_question_text"], how="left")
            merged["annotation"] = merged["annotation"].fillna("very-different")
            examples[(topic_id, rq_rank)] = [
                {r["run_question_text"]: r["annotation"]} for _, r in merged.iterrows()
            ]

    # Build and process tasks
    participant_runs = sorted(set(all_questions["run_tag"].unique()) - set(organizer_runs))
    outputs = []
    tasks = []
    for topic_id, topic_rubrics in sorted(rubrics.items()):
        for question in topic_rubrics:
            rq_rank = int(question["question_id"].split("-")[-1])
            rq_text = question["question_text"]
            for run_tag in participant_runs:
                part = all_questions[
                    (all_questions["topic_id"] == topic_id) & (all_questions["run_tag"] == run_tag)
                ].sort_values("run_question_rank")
                for _, row in part.iterrows():
                    tasks.append((topic_id, rq_rank, rq_text, run_tag, row["run_question_rank"],
                                  row["run_question_text"]))

    for topic_id, rq_rank, rq_text, run_tag, run_q_rank, candidate in tqdm(tasks, desc="auto_question_evaluation"):
        user_input = (
            f"Below is the news article:\n\n"
            f"{json.dumps(articles[topic_id], ensure_ascii=False, indent=2)}\n\n"
            f"This is the target question: {rq_text}\n\n"
            f"Below are some example candidate questions with assessments:\n\n"
            f"{json.dumps(examples[(topic_id, rq_rank)], ensure_ascii=False, indent=2)}\n\n"
            f"Assess the following candidate question:\n\n"
            f'{json.dumps({"candidate_question": candidate}, ensure_ascii=False, indent=2)}\n\n'
            f"Provide your reasoning in a rationale field and your decision in an assessment_decision field."
        )
        reasoning, content = call_llm(system_prompt, user_input, QuestionAssessment.model_json_schema())
        result = QuestionAssessment.model_validate_json(content)
        outputs.append({
            "topic_id": topic_id,
            "rubric_question_rank": rq_rank,
            "run_tag": run_tag,
            "run_question_rank": run_q_rank,
            "assessment": result.assessment_decision,
            "rationale": result.rationale,
        })

    pd.DataFrame(outputs).to_csv(os.path.join(output_folder, "auto_question_assessments.csv"), index=False)


# ── Task: auto_report_evaluation ─────────────────────────────────────────────

def run_auto_report_evaluation(input_folder, output_folder):
    system_prompt = open(os.path.join(SCRIPT_DIR, "system_prompts", "report_judge.txt")).read()
    articles = load_articles()
    rubrics = load_rubrics()
    human_assessments = pd.read_csv(os.path.join(DATA_DIR, "human_assessments", "report_assessments.csv"))

    # Strip references from rubrics (not needed for judging)
    for topic_rubrics in rubrics.values():
        for q in topic_rubrics:
            for a in q["short_answers"]:
                a.pop("references", None)

    # Load participant reports
    reports = []
    for path in sorted(glob.glob(os.path.join(input_folder, "*"))):
        run_tag = os.path.basename(path)
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                topic_id = data["metadata"]["topic_id"]
                report_text = " ".join(s["text"] for s in data["responses"])
                reports.append({"topic_id": topic_id, "run_tag": run_tag, "report": report_text})
    reports = pd.DataFrame(reports)

    # Identify organizer baseline runs (used as few-shot examples, excluded from judging)
    organizer_runs = sorted(set(human_assessments["run_tag"].unique()) & set(reports["run_tag"].unique()))

    # Build few-shot examples from organizer baselines + human assessments
    examples = {}
    for topic_id in sorted(human_assessments["topic_id"].unique()):
        for org_run in organizer_runs:
            report_text = reports[
                (reports["topic_id"] == topic_id) & (reports["run_tag"] == org_run)
            ]["report"].values[0]
            assessments = human_assessments[
                (human_assessments["topic_id"] == topic_id) & (human_assessments["run_tag"] == org_run)
            ].sort_values(by="answer_id", key=lambda x: x.str.extract(r"(\d+)$")[0].astype(int))
            assessments_dict = {r["answer_id"]: r["annotation"] for _, r in assessments.iterrows()}
            examples[(topic_id, org_run)] = {"report": report_text, "assessments": assessments_dict}

    # Process non-organizer reports
    participant_reports = reports[~reports["run_tag"].isin(organizer_runs)].sort_values("topic_id")

    outputs = []
    for _, row in tqdm(participant_reports.iterrows(), total=len(participant_reports), desc="auto_report_evaluation"):
        topic_id, run_tag, report_text = row["topic_id"], row["run_tag"], row["report"]

        example_strs = []
        for i, org_run in enumerate(organizer_runs, 1):
            example_strs.append(
                f"Example {i}:\n\n"
                f"{json.dumps(examples[(topic_id, org_run)], ensure_ascii=False, indent=2)}"
            )

        user_input = (
            f"Below is the news article:\n\n"
            f"{json.dumps(articles[topic_id], ensure_ascii=False, indent=2)}\n\n"
            f"Below is the rubric used to assess reports:\n\n"
            f"{json.dumps(rubrics[topic_id], ensure_ascii=False, indent=2)}\n\n"
            f"Below are example reports and their assessments:\n\n"
            f"{chr(10).join(example_strs)}\n\n"
            f"Below is the report you need to assess, based on the rubric above and given examples. "
            f"Assess whether this report supports, partially supports (partial), contradicts, "
            f"or has no relation (none) to each short answer in the rubric.\n\n"
            f"{json.dumps(report_text, ensure_ascii=False, indent=2)}\n\n"
        )
        reasoning, content = call_llm(system_prompt, user_input,
                                      ReportAssessments.model_json_schema(), schema_name="assessments")
        result = ReportAssessments.model_validate_json(content)
        for a in result.assessments:
            outputs.append({
                "topic_id": topic_id,
                "run_tag": run_tag,
                "answer_id": a.answer_id,
                "assessment": a.assessment_decision,
                "reasoning": a.reasoning,
            })

    pd.DataFrame(outputs).to_csv(os.path.join(output_folder, "auto_report_assessments.csv"), index=False)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAGUN AutoJudge")
    parser.add_argument("--task", required=True,
                        choices=["compound_question_check", "auto_question_evaluation", "auto_report_evaluation"])
    parser.add_argument("--input_folder_path", required=True, help="Folder containing run files")
    parser.add_argument("--output_folder_path", required=True, help="Folder to write output CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_folder_path, exist_ok=True)

    if args.task == "compound_question_check":
        run_compound_question_check(args.input_folder_path, args.output_folder_path)
    elif args.task == "auto_question_evaluation":
        run_auto_question_evaluation(args.input_folder_path, args.output_folder_path)
    elif args.task == "auto_report_evaluation":
        run_auto_report_evaluation(args.input_folder_path, args.output_folder_path)
