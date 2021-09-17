#!/usr/bin/env python3

import os
import shutil
import fileinput

## Fill in Melloddy model best hyperparamters 
hidden_sizes=1200
epochs_lr_steps=(20,10)
dropouts=0.8


# Loop over hyperparameter combinations and edit script
layers_factor=[1, 1.5, 2]
        
for factor in layers_factor:
   
    layer_size = int(int(hidden_sizes)*factor)


    #### Baseline run_sparsechem
    folder_name = f'baseline_{layer_size}'

    os.makedirs(folder_name, exist_ok=True)

    with open("run_sparsechem_baseline.TEMPLATE.sh", "rt") as template:

        filename=os.path.join(folder_name, "run_sparsechem.sh")

        # Open new script file
        with open(filename, "wt") as script:

            # Replace placeholders for hyperparameters
            for line in template:
                script.write(line.replace("HIDDEN_SIZES", str(layer_size)).replace("DROPOUT", str(dropouts)).replace("EPOCHS", str(epochs_lr_steps[0])).replace("LR_STEPS", str(epochs_lr_steps[1])))
                            

               