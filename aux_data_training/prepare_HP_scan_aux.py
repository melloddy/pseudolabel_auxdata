#!/usr/bin/env python3

import os
import shutil
import fileinput

hidden_sizes = '1200'#['50','100','200','400','1000','2000','125 67','250 125','500 250','750 375','1000 500','125 67 32','250 125 67' '500 250 125', '250 250 250', '125 125 125', '500 125 125', '250 67 67', '125 34 34' ]
epochs_lr_steps= (20,10) #[(5,3),(10,5),(20,10)]
dropouts= 0.8 #[0.6,0.4,0.2,0.1,0.]
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
    hidden_name=hidden_sizes.replace(" ", "")
    ppv_npv_name=str(ppv_npv).replace(".","_")
    run_name=f"Run_{num}_epoch_lr_step_{epochs_lr_steps[0]}_{epochs_lr_steps[1]}_drop_{dropouts}_size_{hidden_name}_ppv{ppv_npv_name}_npv{ppv_npv_name}"
    
    # Create script folder and create script
    os.mkdir(f"ppv_npv_scan/{run_name}")	
    filename=f"ppv_npv_scan/{run_name}/run_sparsechem_aux.sh"

    # Read in template
    with open("run_sparsechem_aux.TEMPLATE.sh", "rt") as template:

        # Open new script file
        with open(filename, "wt") as script:

            # Replace placeholders for hyperparameters
            for line in template:
                script.write(line.replace("HIDDEN", hidden_sizes).replace("DROPOUT", str(dropouts)).replace("EPOCHS", str(epochs_lr_steps[0])).replace("LR_STEPS", str(epochs_lr_steps[1])).replace("PPV_NPV", str(ppv_npv).replace(".","_")))
                
                