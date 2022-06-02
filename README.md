# mlr_organocatalyst_fragments
Python scripts for MLR regression fitting accompanying the paper "Harvesting the Fragment-Based Nature of Bifunctional Organocatalysts to Enhance Their Activity.

The scripts construct several MLR regression models based on the contents of the file "DA_Th_Ur_Sq_MLR_all.csv", which must be present in the working directory for execution. Both scripts can be executed directly using python and have minimal dependencies (numpy, pandas, scipy, scikit-learn and matplotlib).

Upon execution, both scripts read the data from the aforementioned file, normalize it with a MaxAbsScaler (i.e. $\frac{x_i}{max(abs(x))}$ ) select the target to regress (either logTOF or DeltaG_3 as per the column headers of the file) and construct models with different choices and number of parameters. The MAEs and regression coefficients of the models are printed, as well as their mathematical expressions.
