import numpy as np
import pandas as pd
from datasets import Dataset
from constants import *

def build_prompt(row):
    return f"""{system_prompt}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{judge_words} Yes
2) {row["negative_example"]}
{judge_words} No
Comment: {row["body"]}
{judge_words}"""

def get_df():
    merge = list()
    if use_train:
        train_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
        train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                                "positive_example_1", "positive_example_2", 
                                "negative_example_1", "negative_example_2"]].copy()
        train_df["positive_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["positive_example_1"], train_df["positive_example_2"])
        train_df["negative_example"] = np.where(np.random.rand(len(train_df)) < 0.5, train_df["negative_example_1"], train_df["negative_example_2"])
        train_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
        merge.append(train_df)
    test_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_dataset = test_dataset.groupby('rule', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"Select {len(test_dataset)} test data")
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule", "subreddit", "positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]].copy()
            body_col = f"{violation_type}_example_{i}"
            other_positive_col = f"{violation_type}_example_{3-i}"
            sub_dataset["body"] = sub_dataset[body_col]
            sub_dataset[f"{violation_type}_example"] = sub_dataset[other_positive_col]
            anti_violation_type = "negative" if violation_type == "positive" else "positive"
            sub_dataset[f"{anti_violation_type}_example"] = np.where(np.random.rand(len(sub_dataset)) < 0.5, sub_dataset[f"{anti_violation_type}_example_1"], sub_dataset[f"{anti_violation_type}_example_2"])
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
            merge.append(sub_dataset)
    return pd.concat(merge, axis=0).drop_duplicates(ignore_index=True)

def build_dataset(df):
    df["prompt"] = df.apply(build_prompt, axis=1)
    columns = ["prompt"]
    if "rule_violation" in df:
        df["completion"] = df["rule_violation"].map({
            1: positive,
            0: negative,})
        columns.append("completion")
    dataset = Dataset.from_pandas(df[columns])
    return dataset
