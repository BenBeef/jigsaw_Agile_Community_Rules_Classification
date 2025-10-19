import pandas as pd
import re

# from rule_augmenter import build_synthetic_rule_map

def url_to_semantics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv") 
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
# def get_dataframe_to_train(data_path):
#     """
#     Load TRAIN only for real comments (labels exist),
#     append ALL examples from train + test,
#     create 3 synthetic rules per original rule,
#     return augmented, de-duplicated, shuffled DataFrame.
#     """
#     train_df = pd.read_csv(f"{data_path}/train.csv")
#     test_df  = pd.read_csv(f"{data_path}/test.csv")

#     # 1. synthetic rule map from **union** of rules (train + test)
#     unique_rules = pd.concat([train_df, test_df])['rule'].unique()
#     rule_map     = build_synthetic_rule_map(unique_rules)

#     # 2. real comments → ONLY from train (have labels)
#     base = train_df[["body", "rule", "subreddit", "rule_violation"]].copy()

#     # 3. helper: augment a dataframe (original + 3 synthetic rules per row)
#     def _augment(df):
#         rows = []
#         for _, r in df.iterrows():
#             rows.append(r.copy())                # keep original
#             for syn_rule in rule_map[r['rule']]: # 3 synthetic variants
#                 rr = r.copy()
#                 rr['rule'] = syn_rule
#                 rows.append(rr)
#         return pd.DataFrame(rows)

#     # 4. collect **all example rows** (pos/neg × 1,2) from **both** files
#     flatten = [base]
#     for df in (train_df, test_df):
#         for vt in ["positive", "negative"]:
#             for i in (1, 2):
#                 col = f"{vt}_example_{i}"
#                 if col not in df.columns:
#                     continue
#                 tmp = df[[col, "rule", "subreddit"]].rename(columns={col: "body"})
#                 tmp["rule_violation"] = 1 if vt == "positive" else 0
#                 tmp = tmp.dropna(subset=["body"])
#                 tmp = tmp[tmp["body"].str.strip().str.len() > 0]
#                 if not tmp.empty:
#                     flatten.append(tmp)

#     # 5. augment the whole pool
#     pool = pd.concat(flatten, ignore_index=True)
#     pool = _augment(pool)          # synthetic rules added here

#     # 6. de-duplicate & shuffle
#     pool = pool.drop_duplicates(subset=["body", "rule", "subreddit"], ignore_index=True)
#     pool = pool.drop_duplicates(subset=["body"], keep="first")
#     return pool.sample(frac=1, random_state=42).reset_index(drop=True)

