import os
import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"
  
    EPOCHS = 3
    LEARNING_RATE = 4e-5  
    
    MAX_LENGTH = 512
    BATCH_SIZE = 12

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def main():
    seed_everything(42)
    training_data_df = get_dataframe_to_train(CFG.data_path)
    # training_data_df, valid_df = train_test_split(full_df,test_size=0.2,stratify=full_df['rule'],random_state=42)
    print(f"Training dataset (from examples only) size: {len(training_data_df)}")

    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")
    
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)
    train_encodings = tokenizer(training_data_df['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",  #这一行加上这个 save_strategy="no"
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['input_text'] = test_df_for_prediction['rule'] + "[SEP]" + test_df_for_prediction['body_with_url']
    
    test_encodings = tokenizer(test_df_for_prediction['input_text'].tolist(), truncation=True, padding=True, max_length=CFG.MAX_LENGTH)
    test_dataset = JigsawDataset(test_encodings)
    
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission_debertaauc.csv", index=False)

if __name__ == "__main__":
    main()

