import dendropy
import utils
import os
import numpy as np


# generic functions for all simulators

def generate_gt_alignments(t, num_gene_trees, length, scale, temp_dir, iqtree_path, num_taxa, pseudo_directory, model):
    gene_trees = []
    alignments = []
    pis_list = []
    gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
        containing_taxon_namespace=t.taxon_namespace,
        num_contained=1)
    for gene_tree in range(num_gene_trees):
        
        os.system('mkdir %s' % temp_dir)

        gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
        gene_trees.append(gene_t)
        
        gene_t.write(path = '%s/temp.tre' % temp_dir, schema='newick')
        
        os.system('%s --alisim %s/temp -t %s/temp.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' % (iqtree_path, temp_dir, temp_dir, model, length, scale))

        char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/temp.phy' % temp_dir), schema="phylip")
        alignments.append(char_matrix)


        # count PIS
        pis = utils.count_pis(char_matrix, num_taxa)
        pis_list.append(pis)
    
        # write gene tree to a file
        gene_t.write(path = '%s/gene_trees/gene%s.tre' % (pseudo_directory, gene_tree), schema = "newick" )
        # write alignment to file
        char_matrix.write(path = '%s/alignments/gene%s.phy' % (pseudo_directory, gene_tree), schema = "phylip" )

        os.system('rm -r %s' % temp_dir)
    return(sum(pis_list)/len(pis_list), np.std(pis_list))

# Classes for simulating data

class Pseudo_Simulator(object):

    def __init__(self, Num_Taxa, Coal_Time, Pseudo_Directory, Num_Gene_Trees, Length, Scale, Temp_Dir, IQTree_Path, Model):
        self.Num_Taxa = Num_Taxa
        self.Coal_Time = Coal_Time
        self.Pseudo_Directory = Pseudo_Directory
        self.Num_Gene_Trees = Num_Gene_Trees
        self.Length = Length
        self.Scale = Scale
        self.Temp_Dir = Temp_Dir
        self.IQTree_Path = IQTree_Path
        self.Model = Model


    def __call__(self):
        
        # generate a random tree
        t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = self.Num_Taxa)
        
        # add branch lengths in coalescent units
        t = utils.add_bl(t, self.Coal_Time)
        
        # write species tree to file
        t.write(path = '%s/species_tree.tre' % self.Pseudo_Directory, schema = "newick", suppress_rooting=True)
        
        # generate num_gene_trees gene trees and alignments
        pis, std_pis = generate_gt_alignments(t, self.Num_Gene_Trees, self.Length, self.Scale, self.Temp_Dir, self.IQTree_Path, self.Num_Taxa, self.Pseudo_Directory, self.Model)
        
        return(pis, std_pis)


class Simulator(object):


    def __init__(self, Temp_Dir, Num_Taxa, IQTree_Path, Model, Input_Match_dict):
        self.Temp_Dir = Temp_Dir
        self.Num_Taxa = Num_Taxa
        self.IQTree_Path = IQTree_Path
        self.Model = Model
        self.Input_Match_dict = Input_Match_dict

    def simulate_matching(self, empirical_lengths, coal_current, scale_current):

        simulated_pis = []
    
        for length in range(len(empirical_lengths)):
    
            os.mkdir(self.Temp_Dir)
    
            # generate a random tree
            t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = self.Num_Taxa)
        
            # add branch lengths in coalescent units
            t = utils.add_bl(t, coal_current)
        
            # g 2 species map
            gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
                containing_taxon_namespace=t.taxon_namespace,
                num_contained=1)
    
            # generate gene t
            gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
            gene_t.write(path = '%s/temp.tre' % self.Temp_Dir, schema='newick')
            
            os.system('%s --alisim %s/temp -t %s/temp.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' % 
                (self.IQTree_Path, self.Temp_Dir, self.Temp_Dir, self.Model, empirical_lengths[length], scale_current))
    
            # read data and calculate pis
            char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/temp.phy' % self.Temp_Dir), schema="phylip")
            pis = utils.count_pis(char_matrix, self.Num_Taxa)
            simulated_pis.append(pis)
    
            os.system('rm -r %s' % self.Temp_Dir)
    
        mean_simulated_pis = sum(simulated_pis)/len(simulated_pis)
        std_simulated_pis = np.std(simulated_pis)
    
        return(mean_simulated_pis, std_simulated_pis)
    
    def simulate_matching_stage2(self, empirical_lengths, max_var, current_coal, scale_current, Batch_Size, taxon_order):

        # generate a random tree
        t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = self.Num_Taxa)
        t = utils.replace_names(tree_string = t, Input_Match_dict = self.Input_Match_dict)

        all_generated_regions = []
        all_generated_pvar = []
    
        for i in range(Batch_Size):
    
            # add branch lengths in coalescent units
            t = utils.add_bl(t, current_coal)
    
            simulated_pis = []
        
            char_matrix_list = []
        
            for length in range(len(empirical_lengths[i])):
                os.mkdir(self.Temp_Dir)
            
            
                # g 2 species map
                gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
                    containing_taxon_namespace=t.taxon_namespace,
                    num_contained=1)
        
                # generate gene t
                gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
                gene_t.write(path = '%s/temp.tre' % self.Temp_Dir, schema='newick')
                
                os.system('%s --alisim %s/temp -t %s/temp.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' % 
                    (self.IQTree_Path, self.Temp_Dir, self.Temp_Dir, self.Model, empirical_lengths[i][length], scale_current))
        
                # read data and calculate pis
                char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/temp.phy' % self.Temp_Dir), schema="phylip")
                pis = utils.count_pis(char_matrix, self.Num_Taxa)
                simulated_pis.append(pis)
                char_matrix_list.append(char_matrix)
        
                os.system('rm -r %s' % self.Temp_Dir)
        
        
            matrix_full, variable_sites = utils.charmatrix_to_nparray(char_matrix_list, max_var, self.Num_Taxa, taxon_order)
            all_generated_regions.append(matrix_full)
            all_generated_pvar.append(variable_sites)
    
    
        all_generated_regions = np.array(all_generated_regions)
        all_generated_pvar = np.array(all_generated_pvar)
    
        return(all_generated_regions, all_generated_pvar)

    def simulate_matching_stage2_training(self, empirical_lengths, coal_current, scale_current, taxon_order, max_var, Batch_Size, t):
    
        all_generated_regions = []
        all_generated_pvar = []
    
        for i in range(Batch_Size):
    
            # add branch lengths in coalescent units
            t = utils.add_bl(t, coal_current)
            t = utils.replace_names(tree_string = t, Input_Match_dict = self.Input_Match_dict)

            simulated_pis = []
        
            char_matrix_list = []
        
            for length in range(len(empirical_lengths[i])):
                os.mkdir(self.Temp_Dir)
            
            
                # g 2 species map
                gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
                    containing_taxon_namespace=t.taxon_namespace,
                    num_contained=1)
        
                # generate gene t
                gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
                gene_t.write(path = '%s/temp.tre' % self.Temp_Dir, schema='newick')
                
                os.system('%s --alisim %s/temp -t %s/temp.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' % 
                    (self.IQTree_Path, self.Temp_Dir, self.Temp_Dir, self.Model, empirical_lengths[i][length], scale_current))
        
                # read data and calculate pis
                char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/temp.phy' % self.Temp_Dir), schema="phylip")
                pis = utils.count_pis(char_matrix, self.Num_Taxa)
                simulated_pis.append(pis)
                char_matrix_list.append(char_matrix)
        
                os.system('rm -r %s' % self.Temp_Dir)
        
        
            matrix_full, variable_sites = utils.charmatrix_to_nparray(char_matrix_list, max_var, self.Num_Taxa, taxon_order)
            all_generated_regions.append(matrix_full)
            all_generated_pvar.append(variable_sites)
    
    
        all_generated_regions = np.array(all_generated_regions)
        all_generated_pvar = np.array(all_generated_pvar)
    
        return(all_generated_regions, all_generated_pvar)
    
