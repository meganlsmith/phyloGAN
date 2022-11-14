"""Propose NNI, TBR, and SPR tree moves."""
"""Bio.Phylo.TreeConstruction.NNITreeSearcher used for NNI. Downloaded from https://github.com/biopython/biopython/blob/master/Bio/Phylo/TreeConstruction.py"""

import random
from Bio import Phylo
import io
import copy
import ete3
import utils

class TreeMove(object):

    def __init__(self, tree_str, temperature, max_trees, Birth_rate):
        self.tree_str = tree_str
        self.temperature = temperature
        self.max_trees = max_trees
        self.Birth_rate = Birth_rate

    def prune(self,t_orig, prune_name):
        """Function to prune node. Adopted from Azouri et al. 2021)"""
        t_cp_p = t_orig.copy()  # the original tree is needed for each iteration
        prune_node_cp = t_cp_p & prune_name     # locate the node in the copied subtree
        assert prune_node_cp.up

        nname = prune_node_cp.up.name
        prune_loc = prune_node_cp
        prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
        t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

        return nname, prune_node_cp, t_cp_p

    def regraft_branch(self, t_cp_p, rgft_node, prune_node_cp, rgft_name, nname):
        '''
        get a tree with the 2 concatenated subtrees
        '''

        new_branch_length = rgft_node.dist /2
        t_temp = ete3.Tree()
        t_temp.add_child(prune_node_cp)
        t_curr = t_cp_p.copy()
        rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree
        rgft_loc = rgft_node_cp.up
        rgft_node_cp.detach()
        t_temp.add_child(rgft_node_cp, dist=new_branch_length)
        t_temp.name = nname
        rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting

        return t_curr

    def spr(self):

        # read tree
        treeobject = ete3.Tree(self.tree_str)

        ## (1) avoid moves to a branch in the pruned subtree
        ## (2) avoid moves to the branch of a sibiling (retaining the same topology)
        ## (3) avoid moves to the branch leading to the parent of the (retaining the same topology)
        ## (4) avoid moves to the root, since trees are treated as rooted


        # get list of 'descendants' to choose node to prune
        count=0
        toprune=[]
        for item in treeobject.iter_descendants(strategy='levelorder', is_leaf_fn=None):
            if item.name=="":
                toname = 'internal_'+str(count)
                item.name=toname

            # will not include root, but also cannot use first node because pruning it doesn't give us different trees
            if not ((count==0) or (len(item.get_leaf_names())==1)):
                # avoid root
                toprune.append(item.name)
            count+=1


    #    # prune descendant
        newtreelist = []
        for prune_node in toprune:
            nname, prunednodecp, prunedtreecp = self.prune(treeobject, prune_node)
            count=0
            for item in prunedtreecp.iter_descendants(strategy='levelorder', is_leaf_fn=None):
                # avoids moves to branch leading to same parent
                if item.name==nname or count==0:
                    # catch cases that would result in identical b/c 2
                    next
                else:
                    new_tree = self.regraft_branch(prunedtreecp, item, prunednodecp, item.name, nname)
                    #rf, max_rf, common_leaves, parts_t1, parts_t2 = new_tree.robinson_foulds(treeobject)
                    result = new_tree.robinson_foulds(treeobject, unrooted_trees = True)
                    match=False
                    if result[0] == 0:
                        match=True
                    else:
                        for atree in newtreelist:
                            result = new_tree.robinson_foulds(atree, unrooted_trees = True)
                            if result[0] == 0:
                                match=True
                    if match==False:
                        newtreelist.append(new_tree)
                count+=1

        newtreestringlist = []
        for newtree in newtreelist:
            newtreestringlist.append(newtree.write(format=9))

        return newtreestringlist

    def get_neighbors(self, tree):
        """Get all neighbor trees of the given tree (PRIVATE).
        Currently only for binary rooted trees.
        FROM: Bio.Phylo.TreeConstruction.NNITreeSearcher
        """
        # make child to parent dict
        parents = {}
        for clade in tree.find_clades():
            if clade != tree.root:
                node_path = tree.get_path(clade)
                # cannot get the parent if the parent is root. Bug?
                if len(node_path) == 1:
                    parents[clade] = tree.root
                else:
                    parents[clade] = node_path[-2]
        neighbors = []
        root_childs = []
        for clade in tree.get_nonterminals(order="level"):
            if clade == tree.root:
                left = clade.clades[0]
                right = clade.clades[1]
                root_childs.append(left)
                root_childs.append(right)
                if not left.is_terminal() and not right.is_terminal():
                    # make changes around the left_left clade
                    # left_left = left.clades[0]
                    left_right = left.clades[1]
                    right_left = right.clades[0]
                    right_right = right.clades[1]
                    # neightbor 1 (left_left + right_right)
                    del left.clades[1]
                    del right.clades[1]
                    left.clades.append(right_right)
                    right.clades.append(left_right)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # neighbor 2 (left_left + right_left)
                    del left.clades[1]
                    del right.clades[0]
                    left.clades.append(right_left)
                    right.clades.append(right_right)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # change back (left_left + left_right)
                    del left.clades[1]
                    del right.clades[0]
                    left.clades.append(left_right)
                    right.clades.insert(0, right_left)
            elif clade in root_childs:
                # skip root child
                continue
            else:
                # method for other clades
                # make changes around the parent clade
                left = clade.clades[0]
                right = clade.clades[1]
                parent = parents[clade]
                if clade == parent.clades[0]:
                    sister = parent.clades[1]
                    # neighbor 1 (parent + right)
                    del parent.clades[1]
                    del clade.clades[1]
                    parent.clades.append(right)
                    clade.clades.append(sister)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # neighbor 2 (parent + left)
                    del parent.clades[1]
                    del clade.clades[0]
                    parent.clades.append(left)
                    clade.clades.append(right)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # change back (parent + sister)
                    del parent.clades[1]
                    del clade.clades[0]
                    parent.clades.append(sister)
                    clade.clades.insert(0, left)
                else:
                    sister = parent.clades[0]
                    # neighbor 1 (parent + right)
                    del parent.clades[0]
                    del clade.clades[1]
                    parent.clades.insert(0, right)
                    clade.clades.append(sister)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # neighbor 2 (parent + left)
                    del parent.clades[0]
                    del clade.clades[0]
                    parent.clades.insert(0, left)
                    clade.clades.append(right)
                    temp_tree = copy.deepcopy(tree)
                    neighbors.append(temp_tree)
                    # change back (parent + sister)
                    del parent.clades[0]
                    del clade.clades[0]
                    parent.clades.insert(0, sister)
                    clade.clades.insert(0, left)
        return neighbors

    def nni(self):
        tree = Phylo.read(io.StringIO(self.tree_str), "newick")
        tree.root_at_midpoint()
        neighbors = self.get_neighbors(tree)
                
        # unroot trees
        derooted_neighbors = []
        for t in neighbors:
            writetree = io.StringIO()
            Phylo.write(t, writetree, format = "newick")
            data = writetree.getvalue()
            current_tree = ete3.Tree(data)
            current_tree.unroot()
            derooted_neighbors.append(current_tree)
            
        
        # remove duplicates

        
        to_remove = []
        for i in range(len(derooted_neighbors) - 1):
            for j in range(i, len(derooted_neighbors)):
                if i != j:
                    if not i in to_remove and not j in to_remove:
                        distance = derooted_neighbors[i].robinson_foulds(derooted_neighbors[j], unrooted_trees = True)[0]
                        if distance == 0:
                            to_remove.append(j)
        to_keep = []
        for i in range(len(derooted_neighbors)):
            if i not in to_remove:
                to_keep.append(derooted_neighbors[i])
        
        
        neighborlist = []
        for eachtree in to_keep:
            writetree = io.StringIO()
            testwrite = eachtree.write()
            neighborlist.append(testwrite)
        return(neighborlist)

    def nni_single(self, current_tree):
        tree = Phylo.read(io.StringIO(current_tree), "newick")
        tree.root_at_midpoint()
        neighbors = self.get_neighbors(tree)
                
        # unroot trees
        derooted_neighbors = []
        for t in neighbors:
            writetree = io.StringIO()
            Phylo.write(t, writetree, format = "newick")
            data = writetree.getvalue()
            current_tree = ete3.Tree(data)
            current_tree.unroot()
            derooted_neighbors.append(current_tree)
            
        
        # remove duplicates

        
        to_remove = []
        for i in range(len(derooted_neighbors) - 1):
            for j in range(i, len(derooted_neighbors)):
                if i != j:
                    if not i in to_remove and not j in to_remove:
                        distance = derooted_neighbors[i].robinson_foulds(derooted_neighbors[j], unrooted_trees = True)[0]
                        if distance == 0:
                            to_remove.append(j)
        to_keep = []
        for i in range(len(derooted_neighbors)):
            if i not in to_remove:
                to_keep.append(derooted_neighbors[i])
        
        
        neighborlist = []
        for eachtree in to_keep:
            writetree = io.StringIO()
            testwrite = eachtree.write()
            neighborlist.append(testwrite)
        neighborlist = random.sample(neighborlist, 1)[0]
        return(neighborlist)


    def choosemove(self):

        probability = random.uniform(0, 1)
        modified_probability = probability * self.temperature

        sprmin = 0.4
        nnimin = 0.0

        if modified_probability > sprmin:
            move="SPR"
        else:
            move="NNI"

        return(move)
    def spr_single(self, current_tree):

        # read tree
        treeobject = ete3.Tree(current_tree)

        ## (1) avoid moves to a branch in the pruned subtree
        ## (2) avoid moves to the branch of a sibiling (retaining the same topology)
        ## (3) avoid moves to the branch leading to the parent of the (retaining the same topology)
        ## (4) avoid moves to the root, since trees are treated as rooted


        # get list of 'descendants' to choose node to prune
        count=0
        toprune=[]
        for item in treeobject.iter_descendants(strategy='levelorder', is_leaf_fn=None):
            if item.name=="":
                toname = 'internal_'+str(count)
                item.name=toname

            # will not include root, but also cannot use first node because pruning it doesn't give us different trees
            if not ((count==0) or (len(item.get_leaf_names())==1)):
                # avoid root
                toprune.append(item.name)
            count+=1


    #    # prune descendant
        newtreelist = []
        for prune_node in toprune:
            nname, prunednodecp, prunedtreecp = self.prune(treeobject, prune_node)
            count=0
            for item in prunedtreecp.iter_descendants(strategy='levelorder', is_leaf_fn=None):
                # avoids moves to branch leading to same parent
                if item.name==nname or count==0:
                    # catch cases that would result in identical b/c 2
                    next
                else:
                    new_tree = self.regraft_branch(prunedtreecp, item, prunednodecp, item.name, nname)
                    #rf, max_rf, common_leaves, parts_t1, parts_t2 = new_tree.robinson_foulds(treeobject)
                    result = new_tree.robinson_foulds(treeobject, unrooted_trees = True)
                    match=False
                    if result[0] == 0:
                        match=True
                    else:
                        for atree in newtreelist:
                            result = new_tree.robinson_foulds(atree, unrooted_trees = True)
                            if result[0] == 0:
                                match=True
                    if match==False:
                        newtreelist.append(new_tree)
                count+=1

        newtreestringlist = []
        for newtree in newtreelist:
            newtreestringlist.append(newtree.write(format=9))
        newtreestringlist = random.sample(newtreestringlist,1)[0]
        return newtreestringlist


    def autowalk(self, num_iter):

        # create an automated walk away from the true truee
        trees = []
        current_tree = open(self.tree_str, 'r').readlines()[0]

        for i in range(num_iter):
            T = utils.temperature(i, num_iter)

            probability = random.uniform(0, 1)
            modified_probability = probability * T

            nochangemin = 0.3
            nnimin = 0.1
            sprmin = 0.0

            if i <= 5:
                newtree = current_tree

            elif probability >= nochangemin:
                newtree = current_tree
            elif probability >= nnimin:
                newtree = self.nni_single(current_tree)
            elif probability >= sprmin:
                newtree = self.spr_single(current_tree)
            
            newtree = utils.add_bl(newtree, self.Birth_rate)

            current_tree = newtree
            trees.append(newtree)
            del newtree

        trees.reverse()
        return(trees)

    def __call__(self):

        # get up to ten trees using moves as determined by probability and temperature
        # return a list of demographies with trees, population sizes, and divergence times

        firstmove = self.choosemove()

        if firstmove == 'NNI':
            trees1 = self.nni()
            trees2 = self.spr()

        elif firstmove == 'SPR':
            trees1 = self.spr()
            trees2 = self.nni()

        treelist = []
        if len(trees1) > self.max_trees:
            trees1 = random.sample(trees1, self.max_trees)
            trees = trees1
        elif len(trees2) > self.max_trees - len(trees1):
            trees2 = random.sample(trees2, self.max_trees - len(trees1))
            trees = trees1 + trees2
        else:
            trees = trees1 + trees2

        # add branch lengths to trees
        treelist = []
        for tree in trees:
            new_tree = utils.add_bl(tree, self.Birth_rate)
            treelist.append(new_tree)
        return treelist
