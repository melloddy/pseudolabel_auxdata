#!/bin/bash

for file in ppv_npv_scan/Run_*/
do
	echo $file
	cd $file
		chmod u+x run_sparsechem_aux.sh
		if type sbatch > /dev/null 2>&1; then
			sbatch submit_aux.sh
		else
			sh submit_aux.sh
		fi
	cd ../../
done

