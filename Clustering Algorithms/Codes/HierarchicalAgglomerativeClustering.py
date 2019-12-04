import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def get_ground_truth(file):
    file = open(file, 'r')
    gene_data = file.readlines()
    ground_truth_list = []
    for i in range(len(gene_data)):
        ground_truth_list.append(int(gene_data[i].split('\t')[1]))
   # print(ground_truth_list)
    return ground_truth_list

def get_gene_cluster_labels(gene_clusters, gene_rows_len):
    gene_label = [0]*gene_rows_len
    cluster = 1
    gene_fin_dict = dict()
    for gene_list in gene_clusters:
        for gene in gene_list:
            gene_label[gene-1] = cluster
            gene_fin_dict.setdefault(cluster, []).append(gene)
        cluster = cluster+1
    #print(gene_fin_dict)
    return gene_label,gene_fin_dict


def get_gene_clusters(file, cluster_count):
    file = open(file, 'r');
    gene_data = file.readlines();
    gene_rows = len(gene_data);
    gene_row_len = len(gene_data[0].split('\t'))
    gene_data_matrix = [[np.float64(x) for x in line.split('\t')] for line in gene_data]
    gene_data_matrix = np.asarray(gene_data_matrix, dtype=float)
    gene_feature_dist = pdist(gene_data_matrix[0:gene_rows, 2:gene_row_len], 'euclidean')
    gene_dist_matrix = squareform(gene_feature_dist)
    gene_cluster_lists = get_clustered_dict(gene_dist_matrix, cluster_count)
    gene_cluster_labels,gene_fin_dict = get_gene_cluster_labels(gene_cluster_lists,gene_rows)
    return gene_cluster_lists,gene_cluster_labels,gene_fin_dict


def get_clustered_dict(gene_matrix, cluster_count):
    clustered_gene_count = -10;  # initialize clustered_gene_count to a arbitary value lower than 0
    gene_id_dict = {}
    counter = 1
    for i in range(len(gene_matrix)):
        gene_id_dict[i] = str(i + 1)

   # print("start-cluster")
    ''' To check pending elements to be clustered until the cluster count is saturated'''
    while clustered_gene_count != (len(gene_matrix) - int(cluster_count) - 1):

        # if condition satisfies , reintialize clustered_gene_count and update the value after each iteration ends
        clustered_gene_count = 0
        for gene_id in gene_id_dict:
            # once an element or a gourp of element  is clustered we set the value of the former point to an arbitary
            # small value and use that conditon to set the value of clustered_gene_count
            if gene_id_dict[gene_id] == "-999":
                clustered_gene_count = clustered_gene_count + 1;
        # get the indexes of non zerp minimum element
        #min_matrix_val_index = np.where(gene_matrix == np.min(gene_matrix[np.nonzero(gene_matrix)]))[0]

        min_matrix_val_index = np.where(gene_matrix==(gene_matrix[gene_matrix>0]).min())[-1]
        row = min_matrix_val_index[0]
        col = min_matrix_val_index[1]
        for i in range(0, len(gene_matrix)):
            gene_matrix[row][i] = min(gene_matrix[row][i], gene_matrix[col][i]);
            gene_matrix[i][row] = min(gene_matrix[i][row], gene_matrix[i][col]);
            gene_matrix[i][col] = gene_matrix[col][i] = 0
        # print(gene_matrix)

        # group the clusters and store back in the dictionary
        grouped_cluster = gene_id_dict[row] + "&::&" + gene_id_dict[col]
        gene_id_dict[row] = grouped_cluster
        gene_id_dict[col] = "-999"
        # print(clustered_gene_count)

    cluster_list = []
    for i in range(len(gene_matrix)):
        single_cluster = []
        if gene_id_dict[i] != '-999':
            single_cluster.append(gene_id_dict[i])
        if len(single_cluster) > 0:
            cluster_list.append(single_cluster)
    fin_clust = []
    for i in range(len(cluster_list)):
        cls = cluster_list[i][0].split('&::&')
        final_sub_cluster = []
        for val in cls:
            final_sub_cluster.append(int(val))
        fin_clust.append(final_sub_cluster)
    return fin_clust



def compute_similarity_coeff(label, ground_truth):
    m11 = 0
    m10 = 0
    m01 = 0
    m00 = 0


    for i in range(0, len(ground_truth)):
        first_ground_truth = ground_truth[i]
        first_cluster_value = label[i]

        for j in range(0, len(ground_truth)):
            second_ground_truth = ground_truth[j]
            second_cluster_value = label[j]
            ground_truth_value = (first_ground_truth == second_ground_truth)
            cluster_similarity_value = (first_cluster_value == second_cluster_value)

            if (ground_truth_value == True and cluster_similarity_value == True):
                m11 += 1
            elif (ground_truth_value == True and cluster_similarity_value == False):
                m10 += 1
            elif (ground_truth_value == False and cluster_similarity_value == True):
                m01 += 1
            elif (ground_truth_value == False and cluster_similarity_value == False):
                m00 += 1

    jacard = float(m11 / (m11 + m01 + m10))
    #     print("jaccard_coefficient: ", jacard_coeff)
    rand = float((m11 + m00) / (m11 + m00 + m10 + m01))
    #     print("rand_index: ", rand_index)
    return jacard,rand

def plot_PCA(file, labels, plot_title):
    file = open(file, 'r');
    gene_data = file.readlines();
    gene_row_len = len(gene_data[0].split('\t'))
    gene_data_matrix = [[np.float64(x) for x in line.split('\t')] for line in gene_data]
    gene_data_matrix = np.asarray(gene_data_matrix, dtype=float)
    data = gene_data_matrix[:, 2:gene_row_len]
    pca = PCA(n_components=2)
    #data = np.matrix(data).T
    principalComp = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principalComp, columns=['PC1', 'PC2'])
    pca_df["label"] = labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    unique_list = []
    for x in labels:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    colors = ['r', 'g', 'b']
    #print(f'unq {unique_list}')
    for target, color in zip(unique_list, colors):
        indicesToKeep = pca_df['label'] == target
        ax.scatter(pca_df.loc[indicesToKeep, 'PC1']
                   , pca_df.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50)
    ax.legend(unique_list)
    ax.set_title(plot_title)
    ax.grid()
    plt.show()





filelist = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if ".txt" in str(f):
        if not "result" in str(f) and not "DBSCAN" in str(f) and not "new_dataset_1" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for clustering')
print(filelist)


# print(cluster_count)
i = 0;
for file in filelist:
    print("Performing HAC on file \"" + file+"\"")
    cluster_count = input("Enter Desired Number of Cluster: ")
    ground_truth_list = get_ground_truth(file)
    gene_clusters,gene_labels,gene_fin_dict = get_gene_clusters(file, cluster_count)
    #print(f' before {np.unique(gene_labels)}')
    for i in range(len(gene_labels)):
        gene_labels[i] += 1
    #print(f'after {np.unique(gene_labels)}')
    cluster_no = 1
    nf_name = file.replace(".txt","")
    plot_PCA(file,gene_labels,"Scatter Plot for HAC on "+nf_name)
    jaccard, rand = compute_similarity_coeff(gene_labels,ground_truth_list)
   # print(jaccard)
   # print(rand)
    f = open(nf_name+"_AHC_result.txt","w+")
    for c in gene_fin_dict:
        f.write(f'cluster: {c}')
        f.write("\n\n")
        f.write(str(gene_fin_dict[c]))
        f.write("\n\n\n\n")
        cluster_no = cluster_no+1;
    f.write("\n\n")
    f.write(f"Jaccard Coefficient: {jaccard}")

    f.write("\n\n")
    f.write(f"Rand Coefficient: {rand}")
    f.close()
    print("Clustering Complete")
    print("Results written in "+nf_name+"_AHC_result.txt file")
    print("---------------------------------------------")
print("HAC COMPLETE")
