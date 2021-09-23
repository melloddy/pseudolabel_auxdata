#!/bin/bash

for file in ppv_npv_scan_*/Run_*/
do
	echo $file
	cd $file
		chmod u+x run_sparsechem.sh
		if type sbatch > /dev/null 2>&1; then
			sbatch submit.sh
		else
			sh submit.sh
		fi
	cd ../../
done

for file in baseline_*/
do
	echo $file
	cd $file
		chmod u+x run_sparsechem.sh
		if type sbatch > /dev/null 2>&1; then
			sbatch submit.sh
		else
			sh submit.sh
		fi
	cd ../
done
