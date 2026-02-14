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

def load_rubrics():
    rows = []
    for path in sorted(glob.glob(os.path.join(RUBRICS_DIR, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        topic_id = data["topic_id"]
        for q in data["rubrics"]:
            for a in q["short_answers"]:
                rows.append({
                    "topic_id": topic_id,
                    "rubric_question_rank": int(q["question_id"].split("-")[-1]),
                    "answer_id": a["answer_id"],
                    "question_importance": q["importance"],
                })
    df = pd.DataFrame(rows)
    df["question_score"] = df["question_importance"].map(IMPORTANCE_MAP)
    return df


# ── Question generation scoring ──────────────────────────────────────────────

def score_question_generation(assessments, compound_check, rubrics, output_dir, prefix):
    # Map assessment labels to scores
    assessments["score"] = assessments["annotation"].map(
        {"very-similar": 1, "similar": 0.5, "different": 0, "very-different": 0}
    )

    # Apply compound question penalty: compound questions get zero credit
    assessments = assessments.merge(
        compound_check[["topic_id", "run_tag", "run_question_rank", "auto_compound_question_assessment"]],
        on=["topic_id", "run_tag", "run_question_rank"], how="left",
    )
    assessments["score"] = (assessments["score"] *
                            assessments["auto_compound_question_assessment"].map({"compound": 0, "not-compound": 1.0}))

    # Merge rubric importance weights
    rubrics = rubrics[["topic_id", "rubric_question_rank", "question_score"]].drop_duplicates(keep="first")
    assessments = assessments.merge(rubrics, on=["topic_id", "rubric_question_rank"], how="left")
    assessments["score"] = assessments["score"] * assessments["question_score"]

    # Compute per-topic, per-run scores
    results = []
    if assessments.isna().any().any():
        raise "Missing values detected."
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
    assessments = assessments.merge(rubric_answers, on=["topic_id", "answer_id"], how="left")

    # Compute per-topic, per-run scores
    results = []
    if assessments.isna().any().any():
        raise "Missing values detected."
    for topic_id in sorted(assessments["topic_id"].unique()):
        rubric_topic = rubric_answers[rubric_answers["topic_id"] == topic_id]
        max_score = rubric_topic.drop_duplicates(subset=["rubric_question_rank"])["question_score"].sum()
        for run_tag in sorted(assessments[assessments["topic_id"] == topic_id]["run_tag"].unique()):
            part = assessments[(assessments["topic_id"] == topic_id) & (assessments["run_tag"] == run_tag)]
            supportive_total, contradictory_total = 0.0, 0.0
            for qid in sorted(part["rubric_question_rank"].unique()):
                q_part = part[part["rubric_question_rank"] == qid]
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
    parser = argparse.ArgumentParser(description="DRAGUN Scoring with Human or Automatic Assessments")
    parser.add_argument("--task", required=True,
                        choices=["question_generation_evaluation", "report_generation_evaluation"])
    parser.add_argument("--type", required=True, choices=["human", "auto"])
    parser.add_argument("--assessment_input", required=True,
                        help="CSV file containing human or automatic assessments")
    parser.add_argument("--compound_check_input", required=False,
                        help="CSV file containing compound check assessments (required for question evaluation)")
    parser.add_argument("--output", required=True, help="Folder to write result CSVs")
    args = parser.parse_args()

    # Conditional validation: compound_check_input is required for question_generation_evaluation
    if args.task == "question_generation_evaluation" and args.compound_check_input is None:
        parser.error("--compound_check_input is required when --task is question_generation_evaluation")

    os.makedirs(args.output, exist_ok=True)
    rubrics = load_rubrics()

    if args.task == "question_generation_evaluation":
        assessments = pd.read_csv(args.assessment_input)
        if args.type == 'auto':
            assessments = assessments.rename(columns={"auto_assessment": "annotation"})
        compound = pd.read_csv(args.compound_check_input)
        score_question_generation(assessments, compound, rubrics, args.output, args.type)

    elif args.task == "report_generation_evaluation":
        assessments = pd.read_csv(args.assessment_input)
        if args.type == 'auto':
            assessments = assessments.rename(columns={"auto_assessment": "annotation"})
        score_report_generation(assessments, rubrics, args.output, args.type)
