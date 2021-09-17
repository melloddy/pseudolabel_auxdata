import os
import pandas as pd

weights = [0.02, 0.30]

original_weights_df = pd.read_csv('./baseline_plus_aux_data/matrices/cls/cls_weights.csv')
modified_weights_df = original_weights_df.copy()
modified_weights_df.to_csv(f'./baseline_plus_aux_data/matrices/cls/cls_weights_0_1.csv', index=False)
for weight in weights:
    weight_name = str(weight).replace('.','_')
    modified_weights_df.loc[modified_weights_df.task_type == 'AUX_HTS', 'training_weight'] = weight
    modified_weights_df.to_csv(f'./baseline_plus_aux_data/matrices/cls/cls_weights_{weight_name}.csv', index=False)
    