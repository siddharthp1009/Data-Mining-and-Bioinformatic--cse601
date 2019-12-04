import copy
import itertools
import operator
import os
import sys
import numpy as np
import pydotplus
from sklearn import tree
import collections
from graphviz import Source
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image


class TreeNode(object):

    left = None
    right = None
    column = None
    value = None
    split_value= None
    tree_ = None

    def __init__(self, split_value, left, right, column, value):

        self.left = left
        self.right = right
        self.column = column
        self.value = value
        self.split_value = split_value


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
    return train_data,test_data

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



def get_column_for_split(data,str_index):
    split_val_rows = [999999999 for i in range(len(data[0]) - 1)]
    gini_val_rows = [999999999 for i in range(len(data[0]) - 1)]
    for col in range(len(data[0])-1):
        if(col not in str_index):
            update_with_digits(data,col,split_val_rows,gini_val_rows)
        else:
            update_with_alpha(data,col,split_val_rows,gini_val_rows)
    gini_val_rows = np.array(gini_val_rows)
    min_index = -1
    min = sys.maxsize
    for ind in range(len(gini_val_rows)):
        if gini_val_rows[ind] <= min:
            min = gini_val_rows[ind]
            min_index = ind
    return  min_index,split_val_rows[min_index]


def split_on_value(data, col, split_value):
    left = []
    right = []
    for i in range(len(data)):
        current_val = data[i][col]
        if type(split_value) == list:
            if current_val in split_value:
                right.append(data[i])
            else:
                left.append(data[i])
        else:
            if current_val >= split_value:
                right.append(data[i])
            else:
                left.append(data[i])
    return left,right



def build_branches(col_ind, split_value,data,str_index,lev):
    temp_node = TreeNode(split_value,None,None,col_ind,None)
    left_tree, right_tree = split_on_value(data,col_ind,split_value)
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
        temp_node.left = initiate_tree_build(left_tree,str_index,1)
        temp_node.right = initiate_tree_build(right_tree,str_index,2)
        return temp_node




def initiate_tree_build(data_matrix,str_index,lev):
    class_info = []
    for i in data_matrix:
        class_info.append(i[len(data_matrix[0])-1])
    class_info_set = set(class_info)
    if(len(class_info_set)>1):
        data_matrix = np.asarray(data_matrix)
        data_matrix = data_matrix.astype(np.float)
        col_ind,split_value  = get_column_for_split(data_matrix,str_index)
        tree_node = build_branches(col_ind,split_value,data_matrix,str_index,lev)
        return tree_node
    else:
        n_val = class_info_set.pop()
        return TreeNode(None, None, None,None,n_val)

def print_tree(root_node,data_encoder):
    call(['dot', '-Tpng', 'diag.dot', '-o', 'decision_tree.png', '-Gdpi=600'])

    Image(filename='decision_tree.png')



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
    data_encoder = {}
    for i in range(len(data_matrix[0])):
        if (data_matrix[0][i].isalpha()):
            str_index.append(i);
        current_col = data_matrix[:,i]
        current_col = np.unique(np.sort(current_col))
        index = 0
        for value in current_col:
            data_encoder[value] = index
            index = index+1
        for j in range(len(data_matrix)):
            data_matrix[j][i] = data_encoder[data_matrix[j][i]]

    root_node = initiate_tree_build(data_matrix,str_index,-1)
    print_tree(root_node,data_encoder)
    print("dfsdfgdsf")




filelist = []

print(os.listdir(os.getcwd()+"/DecisionTreeData"))
path =os.getcwd()+"/DecisionTreeData/"
files = os.listdir(os.getcwd()+"/DecisionTreeData")
for f in files:
    if ".txt" in str(f) and "dataset" in str(f):
        if not "result" in str(f) and not "new_dataset" in str(f):
            filelist.append(f);

print(f'{len(filelist)} files detected for Decision Tree')
print(filelist)

i=0;
for file in filelist:
    print("working on file " + file)
    nf_name = file.replace(".txt", "")
    initate_algo(path+file)



# brew install graphviz
# pip install -U pydotplus