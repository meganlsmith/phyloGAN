import os
import utils
import random
import numpy as np
import itertools
import dendropy
import string

class Simulator(object):

    def __init__(self, IQ_Tree, Model, Chunks, Length, Input_Alignments, Input_Lengths, Input_Order, Input_Match_dict, Temp):
        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Length = Length
        self.Input_Alignments = Input_Alignments
        self.Input_Lengths = Input_Lengths
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp

    def simulateScaleCoal(self, scale, coal):
        """Simulate data in IQTree under some lambda and a random tree topology."""

        # list to hold all results across reps
        all_sites = []


        # do the simulation
        empirical_samples = []
        for rep in range(self.Chunks):
            current_length = 0
            lengths = []
            sampled = []
            while current_length < self.Length:
                new_length_sampled = random.sample(range(len(self.Input_Lengths)),1)[0]
                sampled.append(new_length_sampled)
                new_length = self.Input_Lengths[new_length_sampled]
                lengths.append(new_length)
                current_length += new_length
            final_length = lengths[-1] - (current_length - self.Length)
            lengths[-1] = final_length

            empirical_samples.append(sampled)
            os.system('mkdir %s' % self.Temp)

            # generate a random tree
            t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = len(self.Input_Order))
            # add branch lengths in coalescent units
            t = utils.add_bl_coal(t, coal)
            # g 2 species map
            gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
                containing_taxon_namespace=t.taxon_namespace,
                num_contained=1)

            path_list = ['%s/temp_%s.phy' % (self.Temp, x) for x in range(len(lengths))]


            for the_length in range(len(lengths)):

                # generate gene t
                gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
                gene_t.write(path = '%s/temp_%s.tre' % (self.Temp, str(the_length)), schema='newick')

                os.system('%s --alisim %s/temp_%s -t %s/temp_%s.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' %
                    (self.IQ_Tree, self.Temp, str(the_length), self.Temp, str(the_length), self.Model, lengths[the_length], scale))

            # CONCATENATE ALIGNMENTS
            char_matrix = dendropy.DnaCharacterMatrix.concatenate_from_paths(paths=path_list, schema="phylip")

            # remove temporary files
            os.system('rm -r %s' % self.Temp)

            # calculate proportion invariant
            pinv = self.countPinvIQTreeConcat(char_matrix)
            all_sites.append(pinv)
        
        return all_sites, empirical_samples

    def simulateCoalTree(self, coal):
        """Simulate data in IQTree under some lambda and a random tree topology."""

        t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = len(self.Input_Order))
        # add branch lengths in coalescent units
        t = utils.add_bl_coal(t, coal)
        thetree = utils.replace_names(tree_string = t.as_string(schema="newick"), Input_Match_dict = self.Input_Match_dict)
        thetree = thetree.split('[&R] ')[1]
        return(thetree)

    def simulateStartTree(self, coal):
        """Simulate data in IQTree under some lambda and a random tree topology."""
        # generate a random tree
        t = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = len(self.Input_Order))
        # add branch lengths in coalescent units
        t = utils.add_bl_coal(t, coal) 
        print(self.Input_Match_dict)
        thetree = utils.replace_names(tree_string = t.as_string(schema="newick"), Input_Match_dict = self.Input_Match_dict)
        thetree = thetree.split('[&R] ')[1]
        return(thetree)


    def simulateonTree(self, treename, maxSNPs, coal, scale):
        """Simulate data in IQTree under some lambda and a given tree topology."""

        # list to hold all results across reps
        all_generated_regions = []
        all_generated_pinv = []

        # do the simulation
        empirical_samples = []
        for rep in range(self.Chunks):
            current_length = 0
            lengths = []
            sampled = []
            while current_length < self.Length:
                new_length_sampled = random.sample(range(len(self.Input_Lengths)),1)[0]
                sampled.append(new_length_sampled)
                new_length = self.Input_Lengths[new_length_sampled]
                lengths.append(new_length)
                current_length += new_length
            final_length = lengths[-1] - (current_length - self.Length)
            lengths[-1] = final_length

            empirical_samples.append(sampled)

            os.system('mkdir %s' % self.Temp)
            
            # add branch lengths in coalescent units
            t = utils.add_bl_coal(treename, coal)
            # g 2 species map
            gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
                containing_taxon_namespace=t.taxon_namespace,
                num_contained=1)

            path_list = ['%s/temp_%s.phy' % (self.Temp, x) for x in range(len(lengths))]


            for the_length in range(len(lengths)):

                # generate gene t
                gene_t = dendropy.model.coalescent.contained_coalescent_tree(t, gene_to_species_map)
                gene_t.write(path = '%s/temp_%s.tre' % (self.Temp, str(the_length)), schema='newick')

                os.system('%s --alisim %s/temp_%s -t %s/temp_%s.tre -m %s --length %r --branch-scale %r >/dev/null 2>&1' %
                    (self.IQ_Tree, self.Temp, str(the_length), self.Temp, str(the_length), self.Model, lengths[the_length], scale))

            # CONCATENATE ALIGNMENTS
            char_matrix = dendropy.DnaCharacterMatrix.concatenate_from_paths(paths=path_list, schema="phylip")

            # remove temporary files
            os.system('rm -r %s' % self.Temp)

            # convert to np array
            sim_pinv, sim_matrix = self.sequencetonparray_dendropy(char_matrix, char_matrix.taxon_namespace, maxSNPs)

            # add to full array
            all_generated_regions.append(sim_matrix)
            all_generated_pinv.append(sim_pinv)

        all_generated_regions = np.array(all_generated_regions)
        all_generated_pinv = np.array(all_generated_pinv)


        return(all_generated_regions, all_generated_pinv, empirical_samples)


    def countPinvIQTree(self, align):
        """Convert a simulated alignment into a proportion of invariant sites."""

        chunklength = align.get_alignment_length()
        countvarsites = 0
        for i in range(0, chunklength):
            sequence = list(align[:, i])
            if len(set((sequence))) > 1:
                countvarsites+=1
        # calculate proportion invariant sites
        prop_inv = [1- (countvarsites / chunklength)]
        return(prop_inv)

    def countPinvIQTreeConcat(self, align):
        """Convert a simulated alignment into a proportion of invariant sites."""
        chunklength = align.max_sequence_size
        countvarsites = dendropy.calculate.popgenstat.num_segregating_sites(align, ignore_uncertain=True)
        # calculate proportion invariant sites
        prop_inv = [1- (countvarsites / chunklength)]
        return(prop_inv)


    def countPinvChunk(self, samples):
        """Calculate the proportion of invariant sites from a chunk of the empirical data."""
        all_sites = []
        for i in range(len(samples)):
            current_length = 0
            countvarsites = 0
            for j in range(len(samples[i])):
                charmatrix = self.Input_Alignments[samples[i][j]]
                current_length += charmatrix.max_sequence_size
                if current_length <= self.Length:
                    countvarsites += dendropy.calculate.popgenstat.num_segregating_sites(charmatrix, ignore_uncertain = True)
                else:
                    to_sample = charmatrix.max_sequence_size - (current_length - self.Length)
                    extracted = charmatrix.export_character_indices([*range(to_sample)])
                    countvarsites += dendropy.calculate.popgenstat.num_segregating_sites(extracted, ignore_uncertain = True)
            prop_inv = [1- (countvarsites / self.Length)]
            all_sites.append(prop_inv)
        return(all_sites)

    def sequencetonparray(self, align, sporder, maxSNPs):
        """Convert a fasta chunk to a numpy array for downstream operations."""

        matrix_out=np.empty(shape=(0, len(sporder)))

        # find order in which to traverese:
        traverse_order = []
        for x in self.Input_Order:
            traverse_order.append(sporder.index(x))

        chunklength = align.get_alignment_length()
        countvarsites = 0

        for i in range(0, chunklength):
            sequence = list(align[:, i])
            temparray=[]

            if len(set((sequence))) > 1 and countvarsites < maxSNPs:
                for individual in traverse_order:
                    if sequence[individual] == 'A':
                        temparray.append(0)
                    if sequence[individual] == 'T':
                        temparray.append(1)
                    if sequence[individual] == 'C':
                        temparray.append(2)
                    if sequence[individual] == 'G':
                        temparray.append(3)
                countvarsites+=1
                matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)

            elif len(set((sequence))) > 1 and countvarsites >= maxSNPs:
                countvarsites+=1

       # pad with 4 as needed
        while matrix_out.shape[0] < maxSNPs:
            temparray=[4]*len(sporder)
            matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)

        # transpose genotype matrix so individuals: rows, SNPS: columns
        matrix_out = np.transpose(matrix_out)

        combos = list(itertools.combinations(range(matrix_out.shape[0]), 4))
        combo_rows = []
        for i in combos:
            combo_rows.append(matrix_out[i,:])
        matrix_full = np.concatenate(combo_rows)

        #calculate proportion invariant sites
        prop_inv = [1- (countvarsites / chunklength)]
        return(prop_inv, matrix_full)

    def sequencetonparray_dendropy(self, align, sporder, maxSNPs, is_empirical=False):
        """Convert a fasta chunk to a numpy array for downstream operations."""
 
        matrix_out=np.empty(shape=(0, len(sporder)))
        # find order in which to traverese:
        traverse_order = []
        chunklength = align.max_sequence_size
        countvarsites = 0



        for i in range(0, chunklength):
            temparray=[]
            for individual in self.Input_Order:
                if is_empirical == True:
                    bp = str(align['%s' % individual].value_at(i))
                else:
                    bp = str(align['%s_1' % individual].value_at(i))
                if bp == 'A':
                    temparray.append(0)
                elif bp == 'T':
                    temparray.append(1)
                elif bp == 'C':
                    temparray.append(2)
                elif bp == 'G':
                    temparray.append(3)
                else:
                    temparray.append(4)

            if len(set(temparray)) > 1 and countvarsites < maxSNPs:
                countvarsites+=1
                matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)
            elif len(set(temparray)) > 1 and countvarsites >= maxSNPs:
                countvarsites+=1

       # pad with 4 as needed
        while matrix_out.shape[0] < maxSNPs:
            temparray=[4]*len(sporder)
            matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)

        # transpose genotype matrix so individuals: rows, SNPS: columns
        matrix_out = np.transpose(matrix_out)

        combos = list(itertools.combinations(range(matrix_out.shape[0]), 4))
        combo_rows = []
        for i in combos:
            combo_rows.append(matrix_out[i,:])
        matrix_full = np.concatenate(combo_rows)

        #calculate proportion invariant sites
        prop_inv = [1- (countvarsites / chunklength)]
        return(prop_inv, matrix_full)

