#!/usr/bin/env python3

import os
import shutil
import fileinput


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

# Loop over hyperparameter combinations and edit script
i = 0
for ppv_npv in ppv_npv_list:
    i+=1
    num=str(i).zfill(3)

    # Remove spaces in hidden layers (for file name)
    ppv_npv_name=str(ppv_npv).replace(".","_")
    run_name=f"Preds_{num}_ppv{ppv_npv_name}_npv{ppv_npv_name}"
    
    # Create script folder and create script
    os.mkdir(f"ppv_npv_preds/{run_name}")	
    filename=f"ppv_npv_preds/{run_name}/run_sparsechem_aux.sh"

    # Read in template
    with open("run_sparsechem_aux.TEMPLATE.sh", "rt") as template:

        # Open new script file
        with open(filename, "wt") as script:

            # Replace placeholders for hyperparameters
            for line in template:
                script.write(line.replace("PPV_NPV", str(ppv_npv).replace(".","_")))
                
                