import pandas as pd 
import os 
import sys


paths = sys.argv[1:]

T_melloddy = paths[0]
T_all_images = paths[1]
output_dir = paths[2]

T0_melloddy = pd.read_csv(os.path.join(T_melloddy, 'T0.csv'))
T1_melloddy = pd.read_csv(os.path.join(T_melloddy, 'T1.csv'))
T2_all_images = pd.read_csv(os.path.join(T_all_images, 'T2_images.csv'))

### prepare T1
T1_images = T1_melloddy[T1_melloddy.input_compound_id.isin(T2_all_images.input_compound_id)]
T1_images.to_csv(os.path.join(output_dir, 'T1.csv'), index = False)
    
## filtering compounds without input_assay_id
T2_images = T2_all_images[T2_all_images.input_compound_id.isin(T1_images.input_compound_id)]
T2_images.to_csv(os.path.join(output_dir, 'T2.csv'), index = False)

### prepare T0
T0_images = T0_melloddy[T0_melloddy.input_assay_id.isin(T1_images.input_assay_id)]
T0_images.to_csv(os.path.join(output_dir, 'T0.csv'), index = False)
