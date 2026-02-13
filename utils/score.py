import os
import json
import glob
import argparse
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RUBRICS_DIR = os.path.join(DATA_DIR, "human_rubrics")

IMPORTANCE_MAP = {"A: Have to Know": 4, "B: Good to Know": 2, "C: Nice to Know": 1}


# ── Load rubrics ─────────────────────────────────────────────────────────────

def load_rubric_questions():
    """Load rubric questions with importance weights (for question generation scoring)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RUBRICS_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        topic_id = data["topic_id"]
        for q in data["rubrics"]:
            rows.append({
                "topic_id": topic_id,
                "rubric_question_rank": int(q["question_id"].split("-")[-1]),
                "question_importance": q["importance"],
            })
    df = pd.DataFrame(rows)
    df["question_score"] = df["question_importance"].map(IMPORTANCE_MAP)
    return df


def load_rubric_answers():
    """Load rubric answers with importance weights (for report generation scoring)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RUBRICS_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        topic_id = data["topic_id"]
        for q in data["rubrics"]:
            for a in q["short_answers"]:
                rows.append({
                    "topic_id": topic_id,
                    "question_id": q["question_id"],
                    "answer_id": a["answer_id"],
                    "question_importance": q["importance"],
                })
    df = pd.DataFrame(rows)
    df["question_score"] = df["question_importance"].map(IMPORTANCE_MAP)
    return df


# ── Question generation scoring ──────────────────────────────────────────────

def score_question_generation(assessments, compound_check, rubric_questions, output_dir, prefix):
    # Map assessment labels to scores
    assessments["score"] = assessments["annotation"].map(
        {"very-similar": 1, "similar": 0.5, "different": 0, "very-different": 0}
    )

    # Apply compound question penalty: compound questions get zero credit
    compound_check = compound_check.rename(columns={"rank": "run_question_rank"})
    assessments = assessments.merge(
        compound_check[["topic_id", "run_tag", "run_question_rank", "compound_question"]],
        on=["topic_id", "run_tag", "run_question_rank"], how="left",
    )
    assessments["score"] = assessments["score"] * assessments["compound_question"].map({True: 0, False: 1.0})

    # Merge rubric importance weights
    assessments = assessments.merge(rubric_questions, on=["topic_id", "rubric_question_rank"], how="left")
    assessments["score"] = assessments["score"] * assessments["question_score"]

    # Compute per-topic, per-run scores
    results = []
    for topic_id in sorted(assessments["topic_id"].unique()):
        topic = assessments[assessments["topic_id"] == topic_id]
        max_score = topic.drop_duplicates(subset=["rubric_question_rank"])["question_score"].sum()
        for run_tag in sorted(topic["run_tag"].unique()):
            part = topic[topic["run_tag"] == run_tag]
            # For each rubric question, only the best-matching submitted question counts
            per_rq_scores = part.groupby("rubric_question_rank")["score"].max()
            results.append({"run_tag": run_tag, "topic_id": topic_id, "score": per_rq_scores.sum() / max_score})

    per_topic = pd.DataFrame(results)
    per_run = per_topic.groupby("run_tag", as_index=False)["score"].mean().sort_values("score", ascending=False)

    per_topic.to_csv(os.path.join(output_dir, f"{prefix}_question_generation_per_topic_results.csv"), index=False)
    per_run.to_csv(os.path.join(output_dir, f"{prefix}_question_generation_per_run_results.csv"), index=False)
    print(per_run.to_string(index=False))


# ── Report generation scoring ────────────────────────────────────────────────

def score_report_generation(assessments, rubric_answers, output_dir, prefix):
    # Map assessment labels to scores
    assessments["score"] = assessments["annotation"].map(
        {"supports": 1, "partial": 0.5, "contradicts": -1, "none": 0}
    )

    # Merge rubric importance weights
    assessments = assessments.merge(rubric_answers, on=["topic_id", "question_id", "answer_id"], how="left")

    # Compute per-topic, per-run scores
    results = []
    for topic_id in sorted(assessments["topic_id"].unique()):
        rubric_topic = rubric_answers[rubric_answers["topic_id"] == topic_id]
        max_score = rubric_topic.drop_duplicates(subset=["question_id"])["question_score"].sum()
        for run_tag in sorted(assessments[assessments["topic_id"] == topic_id]["run_tag"].unique()):
            part = assessments[(assessments["topic_id"] == topic_id) & (assessments["run_tag"] == run_tag)]
            supportive_total, contradictory_total = 0.0, 0.0
            for qid in sorted(part["question_id"].unique()):
                q_part = part[part["question_id"] == qid]
                w = q_part["question_score"].iloc[0]
                n = len(q_part)
                supportive_total += q_part[q_part["score"] > 0]["score"].sum() / n * w
                contradictory_total += -q_part[q_part["score"] < 0]["score"].sum() / n * w
            results.append({
                "run_tag": run_tag, "topic_id": topic_id,
                "supportive_score": supportive_total / max_score,
                "contradictory_score": contradictory_total / max_score,
            })

    per_topic = pd.DataFrame(results)
    per_run = per_topic.groupby("run_tag", as_index=False).agg(
        supportive_score=("supportive_score", "mean"), contradictory_score=("contradictory_score", "mean")
    ).sort_values("supportive_score", ascending=False)

    per_topic.to_csv(os.path.join(output_dir, f"{prefix}_report_generation_per_topic_results.csv"), index=False)
    per_run.to_csv(os.path.join(output_dir, f"{prefix}_report_generation_per_run_results.csv"), index=False)
    print(per_run.to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAGUN Scoring")
    parser.add_argument("--task", required=True,
                        choices=["question_generation_evaluation", "report_generation_evaluation"])
    parser.add_argument("--type", required=True, choices=["human", "auto"])
    parser.add_argument("--input", required=True, help="Folder containing assessment CSVs")
    parser.add_argument("--output", required=True, help="Folder to write result CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rubric_questions = load_rubric_questions()
    rubric_answers = load_rubric_answers()

    if args.task == "question_generation_evaluation":
        if args.type == "human":
            assessments = pd.read_csv(os.path.join(args.input, "question_assessments.csv"))
            compound = pd.read_csv(os.path.join(args.input, "compound_question_check.csv"))
        else:
            assessments = pd.read_csv(os.path.join(args.input, "auto_question_assessments.csv"))
            assessments.rename(columns={"assessment": "annotation"}, inplace=True)
            compound = pd.read_csv(os.path.join(args.input, "auto_compound_question_check.csv"))
        score_question_generation(assessments, compound, rubric_questions, args.output, args.type)

    elif args.task == "report_generation_evaluation":
        if args.type == "human":
            assessments = pd.read_csv(os.path.join(args.input, "report_assessments.csv"))
        else:
            assessments = pd.read_csv(os.path.join(args.input, "auto_report_assessments.csv"))
            assessments.rename(columns={"assessment": "annotation"}, inplace=True)
        score_report_generation(assessments, rubric_answers, args.output, args.type)
