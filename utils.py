import copy
import random

import numpy as np

from nasbench import api

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = set([CONV1X1, CONV3X3, MAXPOOL3X3])

class EditUtils():
    def __init__(self, nasbench, max_tries=20):
            self.nasbench = nasbench
            self.max_tries = max_tries
        
    def is_valid(self, matrix, nodes):
        return self.nasbench.is_valid(api.ModelSpec( \
                matrix=matrix, \
                ops=nodes, \
            ))

    def is_not_equal(self, matrix1, matrix2):
        return (matrix1 != matrix2).any()
    
    
    def edit_edge(self, matrix, edit_count):
        n = len(matrix)

        indexes = []
        for i in range(n):
            for j in range(i+1, n):
                indexes.append((i,j))
                
        def try_edit(original_matrix, edit_count):
            matrix = copy.deepcopy(original_matrix)
            if edit_count < 1 : return matrix
            
            for _ in range(edit_count):
                # add edge candidate
                not_have = set()
                for idx in indexes:
                    if not matrix[idx] :  not_have.add(idx)
                # delete edge candidate
                rows = set(np.where(np.sum(matrix, axis=1) > 1)[0])
                columns = set(np.where(np.sum(matrix, axis=0) > 1)[0])
                
                deletable = []
                for r in rows:
                    for c in columns:
                        if matrix[(r,c)] : deletable.append((r,c))

                if len(deletable) > 0 and np.sum(matrix) < 9 : edit = random.randint(1,2)
                elif len(deletable) < 1 : edit = 1
                else : edit = 2
                """
                1 : add edge
                2 : delete edge
                """
                if edit == 1:
                    add_idx = random.sample(not_have, 1)[0]
                    matrix[add_idx] = 1

                else :
                    del_idx = random.sample(deletable, 1)[0]
                    matrix[del_idx] = 0

            fake_ops = [INPUT] + [CONV3X3] * (n-2) + [OUTPUT]
            
            if self.is_valid(matrix, fake_ops) and self.is_not_equal(original_matrix, matrix) :
                return matrix
            else : 
                raise Exception("invalid matrix", matrix)

        success = False

        for _ in range(self.max_tries):
            try :
                edited = try_edit(matrix, edit_count)
                success = True
                break
            except Exception as ex:
                print(ex)

        if success : return edited
        else : 
            raise Exception("invalid matrix exception")


    def replace_node(self, original_nodes, edit_count):
        if edit_count > len(original_nodes)-2 : 
            raise Exception("edit count : {} can not be larger than (node counts-2) : {}".format(edit_count, len(original_nodes)-2))
            
        nodes = copy.deepcopy(original_nodes)
        n = len(nodes)
        replace_idxs = random.sample(range(1,n-1), edit_count)
        for idx in replace_idxs:
            new_op = random.sample(OPS - set(nodes[idx]),1)[0]
            nodes[idx] = new_op
        return nodes
    
    def edit_node_only(self, original_nodes, edit_count):
        return self.replace_node(original_nodes, edit_count)
    

    ## only for edit_count < 3
    ## when edit_count >=3 : need node & edge edit 
    def edit_model(self, matrix, nodes, edit_count):
        node_edit_count = random.randint(0, min(len(nodes)-2, edit_count))
        edge_edit_count = edit_count - node_edit_count

        new_nodes = self.edit_node_only(nodes, node_edit_count)
        new_matrix = self.edit_edge(matrix, edge_edit_count)

        return new_matrix, new_nodes
    
    """
    def edit_both(self, original_nodes, original_matrix, edit_count):
        nodes = copy.deepcopy(original_nodes)
        matrix = copy.deepcopy(original_matrix)
        
        if edit_count < 3 :
            return replace_node(nodes, edit_count)
        else :
            not_replace_count = random.randint(0, edit_count // 3)
            replace_count = edit_count - not_replace_count * 3
    """