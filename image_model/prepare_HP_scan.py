#!/usr/bin/env python3

import os
import shutil
import fileinput

hidden_sizes = ['50','100','200','400','1000','2000','125 67','250 125','500 250','750 375','1000 500','125 67 32','250 125 67' '500 250 125', '250 250 250', '125 125 125', '500 125 125', '250 67 67', '125 34 34' ]
epochs_lr_steps=[(5,3),(10,5),(20,10)]
dropouts=[0.6,0.4,0.2,0.1,0.]


# Loop over hyperparameter combinations and edit script
i = 0
for epoch_lr_step in epochs_lr_steps:
    for dropout in dropouts :
        for hidden in hidden_sizes:
            i+=1
            num=str(i).zfill(3)

            # Remove spaces in hidden layers (for file name)
            hidden_name=hidden.replace(" ", "")
            run_name=f"Run_{num}_epoch_lr_step_{epoch_lr_step[0]}_{epoch_lr_step[1]}_drop_{dropout}_size_{hidden_name}"
            # Create script folder and create script
            os.mkdir(f"HP_scan_image_model/{run_name}")	
            filename=f"HP_scan_image_model/{run_name}/run_sparsechem.sh"

            # Read in template
            with open("run_sparsechem.TEMPLATE.sh", "rt") as template:

                # Open new script file
                with open(filename, "wt") as script:

                    # Replace placeholders for hyperparameters
                    for line in template:
                        script.write(line.replace("HIDDEN", hidden).replace("DROPOUT", str(dropout)).replace("EPOCHS", str(epoch_lr_step[0])).replace("LR_STEPS", str(epoch_lr_step[1])))

