**Image dataset preparation:**

1. Locate and create dataset 	
	- compound id 
	- SMILES 
	- continuous *normalized* segmentation features values from software such as acapella or cellprofiler 
2. Find overlap of compounds with the MELLODDY dataset 
3. Create files for sparsechem
   At least two approaches are possible : <br>
		1. Only recreate Y by filtering Y_melloddy and reordering the rows (also the fold vector)<br>
		2. Go through melloddy tuner again with only the compounds with images, then align the Xi with the Y rows from tuner. This is the preferred/recommended approach in order to avoid tasks during training that are populated too low. Additionally, it helps to identify the duplicate and invalid SMILES in the dataset. <br>
The notebooks will expect .npy file format for the X and Y matrices after reordering


**Notebooks :**
		
Step 1 : 
  
1. HP parameter selection after training on step 1 

 
Step 2 : 

1. y sparse generation used for predicting 
   fold 2 predictions for fitting conformal predictors (requires only labeled datapoints)
   
2. Fit conformal predictors<br>
   half of the fold 2 will be used for fitting, half for evaluation 

3. Some stats on the number of tasks considered 

4. y sparse generation for predictions leading to the pseudolabels via conformal predictors 

5. pseudolabels via conformal predictors (all compounds, including non-MELLODDY compounds)

6. generate T1-style pseudolabel file from predictions

7. generate T1-style pseudolabel file from main task labels 

8. combine the two pseudolabel files from the step2_6 and step2_7 in order to generate main+aux T0,T1,T2 files

Step 3 : 

1. creating set of Y main+aux matrices based on the performance in conformal predictors

2. Generate the y-sparse matrix for final inference for evaluation 
	Main tasks + fold 0 to limit the amount of predictions


**Melloddy tuner interactions**

Settings: <br>
- melloddy pipeline, use the 'open' example_parameters and example_key files, provided in the repo ./config subdir 
- scaffold-based folding
- official melloddy_pipeline environment version used in data preparation for Y2


1. Prior to step 1 modelling: 
	   T0, T1, T2 files for the image compounds and the melloddy labels. <br>
	   These contain the melloddy labels for the imaged compounds. <br>
	   This will allow a more compact Y matrix (compared to the full MELLODDY one) where the tasks that do not meet the size quora are filtered out. <br>
	   This will also allow for identifying and removing duplicate and invalid smiles 
2. Prior to step 3 modelling : 
	   T0, T1, T2 files combining the finished pseudolabels with the complete MELLODDY dataset 
	   
	   
**Sparsechem interactions**
- scaffold-based folding
- official melloddy_pipeline environment version used in data preparation for Y2

1. Step 1 modelling 
	- image-based model 
	- To have Sparsechem accept continuous image features as inputs, see 
		https://git.infra.melloddy.eu/wp2/sparsechem/-/issues/30
	- Take care to align the matrices Ximage (from image dataset) and Y(MELLODDY reduced, from Tuner) properly row-wise 
		I've take the approach to 
			- reorder Ximage
			- reorder the fold vector 
			- remove the empty rows for the resulting Y (not strictly needed)
		HP parameters to scan : 

	```
		fold_te = [0]
		fold_va = [2]
		HPs : 
		hidden_sizes = [50,100,200,400,1000,2000,'125 67','250 125','500 250','750 375','1000 500','125 67 32','250 125 67' '500 250 125', '250 250 250', '125 125 125', '500 125 125', '250 67 67', '125 34 34' ]
		epochs_lr_steps=[(5,3),(10,5),(20,10)]
		dropouts=[0.6,0.4,0.2,0.1,0.]
	``` 


2. Step 3 modelling 
	- HPs ? 
	- The outcome will be compared to the baseline models for the complete MELLODDY dataset 
	

3. Predictions 
	- Prior to Step2_2 
		- creating the image-based predictive probabilities for all compounds with images AND occur in MELLODDY (i.e. training set from step 1) : 
	   	- These will be used to fit the conformal predictor framework 
		- using the y_sparse from step2_1 
		- using the reordered Ximage as inputs 
		- on the HP-optimized image model from step 1 
	- Prior to Step2_5 
		- Conformal predictors will be applied to these predictions 
	    - These are predictions for all compounds with images, including those not in MELLODDY 
	    - using the y_sparse from step2_4
		- using the full Ximage as inputs, including non-melloddy compounds 
		- on the HP-optimized image model from step 1 
	- Post modelling in Step 3, for evaluation  : 
		- predictions on the baseline (MELLODDY best model) for fold 0 <br>
		- predictions on the baseline+aux outputted from step2_8 for fold 0 <br>
		

		

