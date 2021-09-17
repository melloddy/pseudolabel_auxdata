#!/usr/bin/env python3

import os
import shutil
import fileinput

## Fill in Melloddy model best hyperparamters 
hidden_sizes=1200
epochs_lr_steps=(20,10)
dropouts=0.8
ppv_npv_list = [0
       ,0.2
       ,0.4
       ,0.5
       ,0.6
       ,0.7
       ,0.8
       ,0.9
       ,0.95
       ,0.99]

weights = [0.1, 0.02, 0.30]

# Loop over hyperparameter combinations and edit script
for weight in weights :
    weight_name = str(weight).replace('.','_')
    
    if weight != 0.1:
        layers_factor = [1]
    else :
        layers_factor=[1, 1.5, 2]
        
    for factor in layers_factor:
        i = 0
        layer_size = int(int(hidden_sizes)*factor)
        
        
        if factor == 1 and weight == 0.1:
            repeat = 5
        else:
            repeat = 1
            

        #### Aux run_sparsechem
        
        while repeat>0:
            repeat -= 1
            for ppv_npv in ppv_npv_list:
                i+=1
                num=str(i).zfill(3)

                # Remove spaces in hidden layers (for file name)
                ppv_npv_name=str(ppv_npv).replace(".","_")
                run_name=f"Run_{num}_ppv{ppv_npv_name}_npv{ppv_npv_name}"

                # Create script folder and create script
                folder_name = f"ppv_npv_scan_{layer_size}_{weight_name}/{run_name}"
                os.makedirs(folder_name, exist_ok=True)

                # Read in template
                with open("run_sparsechem_aux.TEMPLATE.sh", "rt") as template:
                    filename=os.path.join(folder_name, "run_sparsechem.sh")

                    # Open new script file
                    with open(filename, "wt") as script:

                        # Replace placeholders for hyperparameters
                        for line in template:
                            script.write(line.replace("HIDDEN_SIZES", str(layer_size)).replace("DROPOUT", str(dropouts)).replace("EPOCHS", str(epochs_lr_steps[0])).replace("LR_STEPS", str(epochs_lr_steps[1])).replace("PPV_NPV", str(ppv_npv).replace(".","_")).replace('WEIGHT', weight_name))
                            
               
