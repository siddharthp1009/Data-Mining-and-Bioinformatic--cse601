import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA




def get_k_clusters(k_value, break_count, centroids, adjusted_matrix):
    size_of_label_list = adjusted_matrix.shape[0]
    label_list = [-1 for i in range(size_of_label_list)]
    label_list = np.asarray(label_list)

    centroids_old = np.ones(centroids.shape) * -1
    centroids_points_indices_dict = dict()

    iterations = 0
    centroids_equal = False
    while (centroids_equal == False) and (break_count > 0):
        for i in range(0, size_of_label_list):
            norm_in = centroids - adjusted_matrix[i]
            norm_result = []
            for j in range(norm_in.shape[0]):
                total_sum = 0
                square_root = 0
                temp = []
                temp = norm_in[j]
                for k in range(len(temp)):
                    temp[k] = temp[k] * temp[k]
                total_sum = sum(temp)
                square_root = np.sqrt(total_sum)
                norm_result.append(square_root)
            label_list[i] = np.argmin(norm_result)

        np.copyto(centroids_old, centroids)

        for x in range(0, k_value):
            centroids[x, :] = np.mean(adjusted_matrix[np.where(label_list == x)], axis=0)
            centroids_points_indices_dict[x] = np.asarray(np.where(label_list == x)) + 1

        centroids_equal = np.array_equal(centroids, centroids_old)
        iterations = iterations + 1
        break_count = break_count - 1

    centroid_points_by_id = np.zeros(centroids.shape)
    for i in range(centroids.shape[0]):
        centroid_points_by_id[i] = [(feature + 1) for feature in centroids[i]]

    # calculate the SSE
    sse_local = 0
    for l in range(0, k_value):
        sse_local += np.sum(np.linalg.norm(adjusted_matrix[np.where(label_list == l)] - centroids[l, :], axis=1) ** 2)

    return [sse_local, label_list]


def k_means(k_value, break_count, centroids, nf_name):
    sse_min = sys.float_info.max
    sse_min
    sse_local, label_list = get_k_clusters(k_value, break_count, centroids, nf_name)
    if sse_local < sse_min:
        sse_min = sse_local

    return [sse_min, label_list]



def gaussian_kernel(p1,p2,sigma):
    dist_p = np.linalg.norm(p1-p2)
    return np.exp(-dist_p**2/((sigma**2.)))

def calculate_laplacian_matrix(adj_matrix):

    sum_list = []
    for i in range(len(adj_matrix)):
        h = np.sum(adj_matrix[i])
        sum_list.append(h)

    sum_list_arr = np.array(sum_list)
    degree_matrix = np.diag(sum_list_arr)
    laplacian_matrix = np.subtract(degree_matrix,adj_matrix)
    return laplacian_matrix


def perform_spectral_clustering(file,k, max_iterations, nf_name):
    file = open(file, 'r');
    gene_data = file.readlines();
    gene_rows = len(gene_data);
    gene_row_len = len(gene_data[0].split('\t'))
    gene_data_matrix = [[np.float64(x) for x in line.split('\t')] for line in gene_data]
    gene_data_matrix = np.asarray(gene_data_matrix, dtype=float)
    gene_d = gene_data_matrix[0:gene_rows, 2:gene_row_len]
    adjacency = np.zeros((len(gene_d), len(gene_d)))
    for ii in range(len(gene_d)):
        for jj in range(len(gene_d)):
            adjacency[ii, jj] = gaussian_kernel(gene_d[ii, :], gene_d[jj, :], 0.8)

    laplacian_matrix = calculate_laplacian_matrix(adjacency)
    eval,evec = np.linalg.eig(laplacian_matrix);
    eig_vals_sorted = np.sort(eval)
    eig_vecs_sorted = evec[:, eval.argsort()]
   # cluster_count = int(cluster_count)
    maxdiff = 0;
    loc = -1;
    for i in range(eig_vals_sorted.shape[0]-1):
        diff = eig_vals_sorted[i+1] - eig_vals_sorted[i]
        if abs(diff)>maxdiff:
            maxdiff = diff
            loc = i+1
   # print(maxdiff)
   # print(loc)
    eig_vec_for_kmean = eig_vecs_sorted[:,:loc]
    adjusted_matrix = eig_vec_for_kmean
    centroids = adjusted_matrix[initial_centroid_indices, :]
    sse, cluster_id_list = k_means(k, max_iterations, centroids, adjusted_matrix)
   # km = KMeans(n_clusters=int(cluster_count))
   # km_clust = km.fit(eig_vec_for_kmean)
   # print((km.labels_).shape)
    unique_cluster_labels = np.unique(cluster_id_list)
   # print(unique_cluster_labels)
    cluster_dictionary = {}
    for c_no in unique_cluster_labels:
        for i in range(len(cluster_id_list)):
            if(int(cluster_id_list[i]) == int(c_no)):
                cluster_dictionary.setdefault(c_no,[]).append(i+1)
    return cluster_dictionary,cluster_id_list

def plot_PCA(file, labels, plot_title):
  #  print(type(labels))
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




def compute_similarity_coeff(label, ground_truth, nf_name):
    m11 = 0
    m10 = 0
    m01 = 0
    m00 = 0
    for i in range(0, len(ground_truth)):
        first_ground_truth = ground_truth[i];
        first_cluster_value = label[i];

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




filelist = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if ".txt" in str(f):
        if not "result" in str(f) and not "DBSCAN" in str(f) and not "new_dataset_2" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for clustering')
print(filelist)



#cluster_count = input("Enter Desired Cluster Size: ")

i=0;
for file in filelist:
    print("Performing Spectral Clustering on file " + file)

    nf_name = file.replace(".txt", "")
    max_iterations = int(input('Enter Maximum Number of iterations: '))
    c = []
    number_initial_centroids = int(input('Enter number of initial centroids: '))
    for i in range(0, number_initial_centroids):
        ele = int(input(f'Enter Centroid {i + 1}: '))
        c.append(ele)  # adding the element
    print(f'Selected Centroids => {c}')
    initial_centroid_indices = [int(centroid_index) - 1 for centroid_index in c]
   # print(initial_centroid_indices)
    k = len(initial_centroid_indices)
    gene_clusters, gene_cluster_list = perform_spectral_clustering(file,k, max_iterations, nf_name);
  #  sse, cluster_id_list = k_means(k, max_iterations, centroids, nf_name)
    data = np.loadtxt(file, dtype='float')
    ground_truth = [item[1] for item in data]
    jaccard, rand = compute_similarity_coeff(gene_cluster_list, ground_truth, nf_name)
    for i in range(len(gene_cluster_list)):
        gene_cluster_list[i]+= 1
    plot_PCA(file, gene_cluster_list, "Scatter Plot for Spectral Clustering on " + nf_name)
    f = open(nf_name + "_SPECTRAL_result.txt", "w+")
    for  c in gene_clusters:
        f.write(f'cluster: {c}')
        f.write("\n\n")
        f.write(str(gene_clusters[c]))
        f.write("\n\n\n\n")
    f.write("\n\n")
    f.write(f'Jaccard Coefficient: {jaccard}')
    f.write("\n\n")
    f.write(f'Rand Coefficient: {rand}')
    f.close()
    print("Clustering Complete")
    print("Results written in " + nf_name + "_SPECTRAL_result.txt file")
    print("---------------------------------------------")
print("SPECTRAL CLUSTERING COMPLETE")
