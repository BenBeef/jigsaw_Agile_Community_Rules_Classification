!accelerate launch --config_file accelerate_config.yaml train.py
!python inference.py

import pandas as pd
pd.read_csv('/kaggle/working/submission.csv')