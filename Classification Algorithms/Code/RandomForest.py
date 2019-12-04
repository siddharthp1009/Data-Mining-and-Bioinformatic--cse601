import os
import numpy as np
import math
import sys
import copy
from operator import itemgetter
import operator
import itertools
import random

class TreeNode(object):

    left = None
    right = None
    column = None
    value = None
    split_value= None

    def __init__(self, split_value, left, right, column, value):

        self.left = left
        self.right = right
        self.column = column
        self.value = value
        self.split_value = split_value

def get_data(current_fold, data_matrix,data_per_fold,no_of_folds):
    s = -1
    e = -1
    if(current_fold!=no_of_folds-1):
        s = current_fold*data_per_fold
        e = s+data_per_fold
    else:
        s = current_fold*data_per_fold
        e = len(data_matrix)
    train_data = []
    test_data = []
    for i in range(s,e):
        test_data.append(data_matrix[i])
    range_for_test_data = set(range(s,e))
    d= set(range(0,len(data_matrix)))
    range_for_train_data = d-range_for_test_data
    for i in range_for_train_data:
        train_data.append(data_matrix[i])
    range_for_test_data = list(range_for_test_data)
    range_for_train_data = list(range_for_train_data)
    return train_data,test_data,range_for_train_data,range_for_test_data

def get_splits_from_data(data_matrix,start,current_row,matrix_len):
    return data_matrix[range(start,current_row)],data_matrix[range(current_row,matrix_len)]

def unique_class_pairs(split):
    data_class = split[:, -1]
    unique = np.unique(data_class)
    #type(data_class)
    probability_element = {}
    for class_val in unique:
        for val in data_class:
            if (class_val == val):
                if class_val in probability_element:
                    probability_element[class_val] = probability_element[class_val] + 1
                else:
                    probability_element[class_val] = 1
    return probability_element

def calc_gini_index(split):
    if len(split) < 1:
        return 0
    probability_element = unique_class_pairs(split)
    probability_for_each_class_val = []
    data_len  = len(split)
    for key in probability_element:
        probability_for_each_class_val.append(probability_element[key]/data_len)
    probability_for_each_class_val =  np.square(probability_for_each_class_val)
    probability_sum = np.sum(probability_for_each_class_val)
    gini_index = 1-probability_sum
    return gini_index

def get_splits_from_data_for_alpha(data,col,part):
    first = []
    second = []
    for i in range(len(data)):
        added_f_split = False
        for j in part:
            if data[i][col] == j:
                added_f_split = True
                first.append(data[i])
        if not added_f_split:
            second.append(data[i])
    return np.array(first),np.array(second)


def is_binary(val_list):
    return len(val_list)==2


def get_list_from_voting(label_list):
    #label_list_copy = copy.deepcopy(label_list)
    label_list = np.array(label_list)
    result_labels = []
    for i in range(len(label_list[0])):
        current_col = label_list[:,i]
        unique_label = np.unique(current_col)
        if len(unique_label) == 1:
            result_labels.append(unique_label[0])
        else:
            label_dict = {}
            for i in current_col:
                if i in label_dict:
                    label_dict[i] = label_dict[i]+1
                else:
                    label_dict[i] = 1

            max_voted_class = l_node_value = max(label_dict.items(), key=operator.itemgetter(1))[0]
            result_labels.append(max_voted_class)
    return result_labels





def update_with_alpha(data,col,split_val_rows,gini_val_rows):
    min_split = sys.maxsize
    min_gini = sys.maxsize
    unique_val = np.unique(data[:,col])
    p_lst = []
    for i in range(len(unique_val)):
        parts = itertools.combinations(unique_val, i)
        for pt in parts:
            if (len(pt) > 0):
                p_lst.append(list(pt))
    if(len(p_lst)<1): return
    for part in p_lst:
        f_split, s_split = get_splits_from_data_for_alpha(data,col,part)
        gini_f_split = calc_gini_index(f_split)
        gini_s_split = calc_gini_index(s_split)
        gini_curr_splits = (len(f_split)) / len(data) * gini_f_split + (len(s_split)) / len(data) * gini_s_split
        if gini_curr_splits < min_gini:
            min_gini = gini_curr_splits
            min_split = part
        if( is_binary(unique_val)):
            break
    gini_val_rows[col] = min_gini
    split_val_rows[col] = min_split


def mulitple_criteria(split_value):
    if isinstance(split_value,list):
        return True
    return False

def get_label_from_tree(current_root, q):
    if current_root.value is not None:
        return current_root.value
    else:
        if mulitple_criteria(current_root.split_value):
            list_of_vals = current_root.split_value
            if(q[current_root.column] in list_of_vals):
                return get_label_from_tree(current_root.right, q)
            else:
                return get_label_from_tree(current_root.left,q)
        else:
            val = current_root.split_value
            if (q[current_root.column] >= val):
                return get_label_from_tree(current_root.right, q)
            else:
                return get_label_from_tree(current_root.left, q)




def calc_label_for_test(current_root, test_data):
    test_data = np.asarray(test_data)
    label_for_each_entry = []
    for i in range(len(test_data)):
        q  = test_data[i]
        label = get_label_from_tree(current_root,q)
        label_for_each_entry.append(label)
    return label_for_each_entry


def update_with_digits(data,col,split_val_rows,gini_val_rows):
    min_split = sys.maxsize
    min_gini = sys.maxsize
    data_matrix = copy.deepcopy(data)
    sort_by_row_matrix = data_matrix[data_matrix[:,col].argsort()]
    rows_in_data = len(data)
    for i in range(rows_in_data):
        f_split, s_split = get_splits_from_data(sort_by_row_matrix,0,i,rows_in_data)
        gini_f_split = calc_gini_index(f_split)
        gini_s_split = calc_gini_index(s_split)
        gini_curr_splits = (len(f_split))/rows_in_data*gini_f_split + (len(s_split))/rows_in_data*gini_s_split
        if gini_curr_splits < min_gini:
            min_gini = gini_curr_splits
            min_split = sort_by_row_matrix[i][col]
    gini_val_rows[col] = min_gini
    split_val_rows[col] = min_split





def get_column_for_split(data,str_index,vis_col):
    split_val_rows = [999999999 for i in range(len(data[0]) - 1)]
    gini_val_rows = [999999999 for i in range(len(data[0]) - 1)]
    random_feature_len = round(math.sqrt(len(data[0]) - 1))
    random_set_feature = random.sample(range(0,len(data[0])-1),random_feature_len)
    for col in random_set_feature:
        if(col!=str_index):
                update_with_digits(data,col,split_val_rows,gini_val_rows)
        else:
            if (col in vis_col):
                continue
            else:
                update_with_alpha(data,col,split_val_rows,gini_val_rows)
    gini_val_rows = np.array(gini_val_rows)
    min_index = -1
    min = sys.maxsize
    for ind in range(len(gini_val_rows)):
        if gini_val_rows[ind] <= min:
            min = gini_val_rows[ind]
            min_index = ind
    return min_index, split_val_rows[min_index]


def split_on_value(train_data, col, split_value):
    left = []
    right = []
    for i in range(len(train_data)):
        current_val = train_data[i][col]
        if type(split_value) == list:
            if current_val in split_value:
                right.append(train_data[i])
            else:
                left.append(train_data[i])
        else:
            if current_val >= split_value:
                right.append(train_data[i])
            else:
                left.append(train_data[i])
    return left,right





def build_branches(col_ind, split_value,train_data,str_index,vis_cols):
    temp_node = TreeNode(split_value,None,None,col_ind,None)
    left_tree, right_tree = split_on_value(train_data,col_ind,split_value)
    if(len(left_tree))==0:
        right_tree = np.array(right_tree)
        right_class_count_dict = unique_class_pairs(right_tree)
        l_node_value = max(right_class_count_dict.items(), key=operator.itemgetter(1))[0]
        return  TreeNode(None, None, None, None, l_node_value)
    elif(len(right_tree)==0):
        left_tree = np.array(left_tree)
        left_class_count_dict = unique_class_pairs(left_tree)
        r_node_value = max(left_class_count_dict.items(), key=operator.itemgetter(1))[0]
        return TreeNode(None, None, None, None, r_node_value)
    else:
        temp_node.left = initiate_tree_build(left_tree,str_index,vis_cols)
        temp_node.right = initiate_tree_build(right_tree,str_index,vis_cols)
        return temp_node




def initiate_tree_build(train_data,str_index,visited_cols):
    class_info = []
    for i in train_data:
        class_info.append(i[len(train_data[0])-1])
    class_info_set = set(class_info)
    v_col = copy.deepcopy(visited_cols)
    if len(class_info_set) ==1 :
        n_val = class_info_set.pop()
        return TreeNode(None, None, None, None, n_val)
    elif(len(train_data)<4):
        train_data = np.asarray(train_data)
        node_val = unique_class_pairs(train_data)
        max_count_class = max(node_val.items(), key=operator.itemgetter(1))[0]
        return TreeNode(None,None,None,None,max_count_class)
    else:
        if len(v_col)< len(train_data[0])-1:
            train_data = np.asarray(train_data)
            col_ind, split_value = get_column_for_split(train_data, str_index, v_col)
            if split_value == 999999999:
                value = unique_class_pairs(train_data)
                max_count_class = max(value.items(), key=operator.itemgetter(1))[0]
                return TreeNode(None, None, None, None, max_count_class)
            if col_ind in str_index:
                v_col.append(col_ind)
            elif col_ind != len(train_data[0]) - 1:
                v_col.append(999999999)
            tree_node = build_branches(col_ind, split_value, train_data, str_index, v_col)
            return tree_node
        else:
            train_data = np.asarray(train_data)
            node_val = unique_class_pairs(train_data)
            max_count_class = max(node_val.items(), key=operator.itemgetter(1))[0]
            return TreeNode(None, None, None, None, max_count_class)


def calculate_measure(test_dataset, predictions, fold_value):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    test_label = []
    x = len(test_dataset[0])
    for i in range(len(test_dataset)):
        test_label.append(test_dataset[:len(test_dataset)][i][x - 1])

    for i in range(len(test_label)):
        if (test_label[i] == 1 and predictions[i] == 1):
            TP += 1
        elif (test_label[i] == 1 and predictions[i] == 0):
            FN += 1
        elif (test_label[i] == 0 and predictions[i] == 1):
            FP += 1
        elif (test_label[i] == 0 and predictions[i] == 0):
            TN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)

    print("----------------------------------------------------------------")
    print("For Test Dataset from fold", fold_value + 1)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-1: ", f1)
    return accuracy, precision, recall, f1


def initate_algo(file):
    data = open(file)
    data_lines = data.readlines()
    row_len = len(data_lines)
    column_len = len(data_lines[0].split("\t"))
    data_matrix = [[0 for x in range(column_len)] for y in range(row_len)]
    for row in range(row_len):
        for column in range(column_len):
            val = data_lines[row].split("\t")[column].strip('\n')
            data_matrix[row][column] = val
    data_matrix = np.array(data_matrix)
    str_index = []
    data_matrix = data_matrix.astype(np.float)
    no_of_folds = 10  #10
    no_of_trees = 5  #5
    data_per_fold = math.floor(len(data_matrix)/no_of_folds)
    all_test_accuracy = []
    all_test_precision= []
    all_test_recall = []
    all_test_f1_measure = []
    for i in range(no_of_folds):
        train_data, test_data,range_for_train_data,range_for_test_data = get_data(i,data_matrix,data_per_fold,no_of_folds)
        all_roots = []
        for t in range(no_of_trees):
            train_data_random = data_matrix[ np.random.choice(range_for_train_data, len(train_data),
                                                 replace=True)]
            root_node = initiate_tree_build(train_data_random,str_index,[])
            all_roots.append(root_node)

        all_labels_for_current_test = []
        for e_root in all_roots:
            result_class_on_test = calc_label_for_test(e_root,test_data)
            all_labels_for_current_test.append(result_class_on_test)

        all_labels_for_current_test = np.array(all_labels_for_current_test, dtype=np.float)
        final_class_labels = get_list_from_voting(all_labels_for_current_test)
        current_accuracy, current_precision, current_recall, current_f1_measure = calculate_measure(test_data,final_class_labels,i)
        all_test_accuracy.append(current_accuracy)
        all_test_precision.append(current_precision)
        all_test_recall.append(current_recall)
        all_test_f1_measure.append(current_f1_measure)
    print("----------- FINAL METRICS -----------")
    fin_accuracy = np.sum(all_test_accuracy) / len(all_test_accuracy)
    fin_precision = np.sum(all_test_precision) / len(all_test_precision)
    fin_recall = np.sum(all_test_recall) / len(all_test_recall)
    fin_f1_measure = np.sum(all_test_f1_measure) / len(all_test_f1_measure)
    print("Accuracy: ", fin_accuracy, "\nPrecision: ", fin_precision, "\nRecall: ", fin_recall,
          "\nF1-measure: ", fin_f1_measure)






filelist = []
print(os.listdir(os.getcwd()+"/RandomForestData"))
path =os.getcwd()+"/RandomForestData/"
files = os.listdir(os.getcwd()+"/RandomForestData")
for f in files:
    if ".txt" in str(f) and "dataset" in str(f):
        if not "result" in str(f) and not "new_dataset" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for Random Forest')
print(filelist)

i=0;
for file in filelist:
    print("working on file " + file)
    nf_name = file.replace(".txt", "")
    initate_algo(path+file)


