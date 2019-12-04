// This is the READ ME File to RUN Heirarchical Clustering

The code is a simple .py file and can be run in pycharm(RECOMMENDED) or any python interpreter

Step 1 : Run the code for HeirarchicalAgglomerativeClustering.py

Step 2 : The code will detect all text files in the current directory and will print something like below;


		3 files detected for clustering
		['new_dataset_2.txt', 'iyer.txt', 'cho.txt']
		Performing HAC on file "new_dataset_2.txt"
		Enter Desired Number of Cluster: 

	Step 2 prompts will appear for every single file

Step 3 : Enter the cluster size and hit enter, once the code executes for the first file, it will plot pca in the display   pane (on pycharm and even anaconda), and the cluster results will be written in a file named "{CURRENT_FILE}_AHC_result.txt" along with the rand and Jaccard coefficient.


	On executing step 3 successfully the below message will be shown

	Clustering Complete
	Results written in new_dataset_2_AHC_result.txt file

Step 4: Repeat Steps 2 and 3 for all text files to be tested and all the results will be found in the same folder with name of the file suffixed with "_AHC_result.txt"

Step 5: Once all files have be run a final "HAC COMPLETE" message will be printed.