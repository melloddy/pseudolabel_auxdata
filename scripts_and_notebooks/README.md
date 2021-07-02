**Images:**

1. Locate and create dataset 	
	- compound id 
	- SMILES 
	- continuous normalized segmentation features values from software such as acapella or cellprofiler 
2. Find overlap of compounds with the MELLODDY dataset 
3. Create files for sparsechem
   At least two approaches are possible : 
		1. Only recreate Y by filtering Y_melloddy and reordering the rows (also the fold vector)
		2. Go through melloddy tuner again with only the compounds with images, then align the Xi with the Y rows from tuner 

**General :**
		
Step 1 : 
- Train with Sparsechem using continuous features as input using <br>
```
	fold_te = [0]
	fold_va = [2]
	HPs : 
	hidden_sizes = [50,100,200,400,1000,2000,'125 67','250 125','500 250','750 375','1000 500','125 67 32','250 125 67' '500 250 125', '250 250 250', '125 125 125', '500 125 125', '250 67 67', '125 34 34' ]
	epochs_lr_steps=[(5,3),(10,5),(20,10)]
	dropouts=[0.6,0.4,0.2,0.1,0.]
``` 
This requires scaffold network folding. 
	
  Sparsechem will require a modification in order to accept continuous inputs : 
  https://git.infra.melloddy.eu/wp2/sparsechem/-/issues/30
  
 
Step 2 : 

1. y sparse generation used for predicting 
   fold 2 predictions for fitting conformal predictors (requires only labeled datapoints)
   
2. Fit conformal predictors
   half of the fold 2 will be used for fitting, half for evaluation 

3. Some stats on the number of tasks considered 

4. y sparse generation for predictions leading to the pseudolabels via conformal predictors 

5. pseudolabels via conformal predictors (all compounds)

6. generate T1-style pseudolabel file from predictions

7. generate T1-style pseudolabel file from main task labels 

8. generate aux+main T0,T1,T2 

Step 3 : 

1. creating set of Y main+aux matrices based on the performance in conformal predictors

2. Generate the y-sparse matrix for final inference for evaluation 
	Main tasks + fold 0 to limit the amount of predictions


		
		

