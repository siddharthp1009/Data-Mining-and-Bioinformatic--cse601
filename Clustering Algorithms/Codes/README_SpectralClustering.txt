// This is the READ ME File to RUN Spectral Clustering

The code is a simple .py file and can be run in pycharm(RECOMMENDED) or any python interpreter

Step 1 : Run the code for SpectralClustering.py

Step 2 : The code will detect all text files in the current directory and will print something like below;


	Step 2.1 : 		3 files detected for clustering
					['new_dataset_1.txt', 'iyer.txt', 'cho.txt']
					Performing Spectral Clustering on file new_dataset_1.txt
					Enter Maximum Number of iterations: 

					NOTE : Since we need to perform kmeans we take number of iterations, number of clusters and inital centroids as input

	Step 2.2 :     After entering the number of interations, below message will be printed

					Enter number of initial centroids: 

	Step 2.3:      After entering the number of  centroids (say 2), below messages will display

					Enter Centroid 1: 4
					Enter Centroid 2: 5		

					Once centroids are entered following message will be displayed

					Selected Centroids => [4, 5]


					Step 2 prompts will appear for every single file

Step 3 : Once the code executes for the first file, it will plot pca in the display   pane (on pycharm and even anaconda), and the cluster results will be written in a file named "{CURRENT_FILE}_SPECTRAL_result.txt" along with the rand and Jaccard coefficient.


	On executing step 3 successfully the below message will be shown

	Clustering Complete
	Results written in new_dataset_1_SPECTRAL_result.txt file

Step 4: Repeat Steps 2 and 3 for all text files to be tested and all the results will be found in the same folder with name of the file suffixed with "_SPECTRAL_result.txt"

Step 5: Once all files have be run a final "SPECTRAL CLUSTERING COMPLETE" message will be printed.



