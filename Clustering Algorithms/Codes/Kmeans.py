import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def get_k_clusters(k_value, break_count, centroids, nf_name):
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

    f = open(nf_name + "_KMeans_result.txt", "w+")
    f.write(f'Total Iterations: {iterations}')
    f.write("\n\n")

    for (key, val) in centroids_points_indices_dict.items():
        f.write(f'cluster: {key+1}')
        f.write("\n\n")
        f.write(str(val.flatten()))
        f.write("\n\n\n\n")
    f.close()
    print("Clustering Complete")
    print("Results written in " + nf_name + "_KMeans_result.txt file")


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

    f = open(nf_name + "_KMeans_result.txt", "a+")
    f.write("\n\n")
    f.write(f'Jaccard Coefficient: {jacard}')
    f.write("\n\n")
    f.write(f'Rand Index: {rand}')


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
        if not "result" in str(f)  and not "DBSCAN" in str(f) and not "new_dataset_2" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for clustering')
print(filelist)
max_iterations = int(input('Enter Maximum Number of iterations: '))
c = []
number_initial_centroids = int(input('Enter number of initial centroids: '))
for i in range(0, number_initial_centroids):
    ele = int(input(f'Enter Centroid {i+1}: '))
    c.append(ele)  # adding the element
print(f'Selected Centroids => {c}')

for file in filelist:
    print("Performing KMeans on file " + file)
    nf_name = file.replace(".txt", "")
    data = np.loadtxt(file, dtype='float')
    adjusted_matrix = data[:, 2:]
    initial_centroid_indices = [int(centroid_index) - 1 for centroid_index in c]
    k = len(initial_centroid_indices)
    centroids = adjusted_matrix[initial_centroid_indices, :]
    sse, cluster_id_list = k_means(k, max_iterations, centroids, nf_name)
    for i in range(len(cluster_id_list)):
        cluster_id_list[i] +=1
    plot_PCA(file, cluster_id_list, "Scatter Plot for KMEANS on " + nf_name)
    ground_truth = [item[1] for item in data]
    compute_similarity_coeff(cluster_id_list, ground_truth, nf_name)
print("KMEANS COMPLETE")
