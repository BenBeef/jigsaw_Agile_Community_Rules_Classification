import pandas as pd
import numpy as np

q = pd.read_csv('submission_deberta.csv')
l = pd.read_csv('submission_qwen3.csv')
m = pd.read_csv('submission_distilroberta.csv')
w = pd.read_csv('submission_qwen14b.csv')
#d = pd.read_csv('submission_distilbert.csv')
#s = pd.read_csv('submission_debertsmall.csv')
a = pd.read_csv('submission_debertaauc.csv')

rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)
rm = m['rule_violation'].rank(method='average') / (len(m)+1)
rw = w['rule_violation'].rank(method='average') / (len(w)+1)
#rd = d['rule_violation'].rank(method='average') / (len(d)+1)
#rs = s['rule_violation'].rank(method='average') / (len(s)+1)
ra = a['rule_violation'].rank(method='average') / (len(a)+1)

blend = 0.5*rq + 0.1*rl + 0.1*rm + 0.1*rw + 0.2*ra # or tune the rank-weights with a tiny grid using OOF
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)

import pandas as pd
pd.read_csv('/kaggle/working/submission.csv')
