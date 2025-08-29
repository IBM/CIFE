import os
import ast
import argparse
import pandas as pd


DROP_COLS = [
    "constraint_wise_presence",
    "constraint_presence_response",
    "response",
    "constraint_adherence_responses",
    "Constraint_adherence",
    "correctness_level",
    "code_correctness_response",
]

def compute_ssr_from_adherence(series: pd.Series) -> pd.Series:

    def ssr(x):
        lst = ast.literal_eval(x) if isinstance(x, str) else x
        return sum(lst)/len(lst) if lst else 0
    return series.apply(ssr)

def count_constraints(entry) -> int:
    """
    Count number of constraints in a 'final_constraints' list or list-string.
    """
    lst = ast.literal_eval(entry) if isinstance(entry, str) else entry
    return len(lst)

def rank_and_save(
    input_folder: str,
    ranked_folder: str,
    ranked_filename: str = "ranked_dataset.jsonl"
) -> str:
    """
    Stage 1: Rank the examples by SSR and constraint count, then save to JSONL.
    Returns the path to the ranked JSONL.
    """
    base_df = None
    ssr_frames = []
    ssr_cols = []

    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(".jsonl"):
            continue

        model_name = os.path.splitext(fname)[0]
        path = os.path.join(input_folder, fname)
        df = pd.read_json(path, lines=True)


        col_ssr = f"ssr_{model_name}"
        ssr = compute_ssr_from_adherence(df["Constraint_adherence"])
        ssr_frames.append(pd.DataFrame({"id": df["id"], col_ssr: ssr}))
        ssr_cols.append(col_ssr)

    
        if base_df is None:
            base_df = df.drop(columns=DROP_COLS, errors="ignore").copy()

 
    master = base_df.copy()
    for ssr_df in ssr_frames:
        master = master.merge(ssr_df, on="id", how="left")


    master["overall_ssr"] = master[ssr_cols].mean(axis=1)
    master["num_constraints"] = master["final_constraints"].apply(count_constraints)


    master = master.sort_values(
        by=["overall_ssr", "num_constraints"],
        ascending=[False, False]
    ).reset_index(drop=True)


    os.makedirs(ranked_folder, exist_ok=True)
    ranked_path = os.path.join(ranked_folder, ranked_filename)
    master.to_json(ranked_path, orient="records", lines=True)
    print(f"Stage 1: Ranked dataset saved to: {ranked_path}")
    return ranked_path

def get_tail_ids(ranked_jsonl: str, drop_top: int) -> set:
    """
    Load the ranked JSONL, drop the first `drop_top` rows,
    and return the set of remaining `id` values.
    """
    ranked = pd.read_json(ranked_jsonl, lines=True)
    tail = ranked.iloc[drop_top:]
    return set(tail["id"].tolist())

def filter_and_save_all(
    original_folder: str,
    tail_ids: set,
    filtered_folder: str
):
    """
    For each JSONL in original_folder, keep only rows whose `id` is in tail_ids,
    and write to filtered_folder with the same filename.
    """
    os.makedirs(filtered_folder, exist_ok=True)
    for fname in sorted(os.listdir(original_folder)):
        if not fname.lower().endswith(".jsonl"):
            continue

        in_path = os.path.join(original_folder, fname)
        df = pd.read_json(in_path, lines=True)
        filtered = df[df["id"].isin(tail_ids)]

        out_path = os.path.join(filtered_folder, fname)
        filtered.to_json(out_path, orient="records", lines=True)
        print(f"Filtered {fname}: {len(df)} → {len(filtered)} rows → {out_path}")



if __name__ == "__main__":
    input_folder = "LLMjudge_outputs_opensource/code_correctness"
    output_folder = "LLMjudge_outputs/filtered_final_data"
    ranked_filename = "ssr_ranked_dataset.jsonl"
    drop_top = 489 


    # ranked_path = rank_and_save(
    #     input_folder=input_folder,
    #     ranked_folder=output_folder,
    #     ranked_filename=ranked_filename
    # )

    ranked_path = "LLMjudge_outputs/filtered_final_data/ssr_ranked_dataset.jsonl"

    tail_ids = get_tail_ids(ranked_path, drop_top)
    filter_and_save_all(
        original_folder=input_folder,
        tail_ids=tail_ids,
        filtered_folder=output_folder
    )
