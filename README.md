<p align="center">
<img src="https://forthebadge.com/images/badges/made-with-python.svg" >
</p>

<h2 align="center">
 About 
 </h2>
Base Structure for Retrain machine learning model with Hyperparameter tuning. Made for Regression problems, Dev can retrain, optimize and find the best model for the problem out of 11 Algorithms and go for the best from it.
 
<h2 align="center"> ⛩ Architecture </h2>
<p align="center">
<img align="center" src="https://github.com/subha996/Retrain-Life-Cycle/blob/main/retrain_lyf1.png" >
</p>

### To do before running
* Go to `./params` folder where multiple YAML files are created, files are named as their algorithms name. dev only changes the hyperparameters he wants to search for every algorithm.
* run the `train.py` file for start the training or can use the pre-build the `flask` API to starting the training.
<p align="center">
<img align="center" src="https://github.com/subha996/Retrain-Life-Cycle/blob/main/train_tune%20(1).png" >
</p>

### Results
* `./Models_Results` folder `csv` file will be create for different cluster, read csv files as `DataFrame`. These tables will contain result of grid-search, manually can observe the metrices for differents algorithms perfomance.
* `./BestParams` will contai multiple folders and files with best hypermarametr for a algorithm for every cluster.
* `./Models` folder will contain saved model file (`pkl`).

These results are stored separately for manually observing the results, although the process will be fully automatic and it will use the best parameter and best algorithm found for each cluster by itself.

##### Contributor <img src="https://media3.giphy.com/media/1wrgDc6j07hAlM7Jml/giphy.gif?cid=790b7611e3af35beee6df1266c31edcabc53abfbbb82854c&rid=giphy.gif&ct=g" width="30"> 

[Subhabrata Nath](https://www.linkedin.com/in/subhabrata-nath-181375115/)
