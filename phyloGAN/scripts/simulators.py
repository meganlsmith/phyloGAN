import os
import utils
import random
import numpy as np
import itertools

class Simulator(object):

    def __init__(self, IQ_Tree, Model, Chunks, Length, Input_Alignment, Input_Order, Input_Match_dict, Temp):
        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Length = Length
        self.Input_Alignment = Input_Alignment
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp

    def simulateLambda(self, birthRate):
        """Simulate data in IQTree under some lambda and a random tree topology."""

        # list to hold all results across reps
        all_sites = []


        # do the simulation
        for rep in range(self.Chunks):

            os.system('mkdir %s' % self.Temp)
            os.system('%s --alisim %s/temp_tree -t RANDOM{bd{%r/0}/%r} -m %s --length %r --redo >/dev/null 2>&1' % (self.IQ_Tree, self.Temp, birthRate, len(self.Input_Order), self.Model, self.Length))

            thetree = open('%s/temp_tree.treefile' % self.Temp, 'r').readlines()[0].strip()

            thetree = utils.add_bl(thetree, birthRate)

            thetree = utils.replace_names(tree_string = thetree, Input_Match_dict = self.Input_Match_dict)

            newtree = open('%s/temp_tree.treefile' % self.Temp, 'w')
            newtree.write(thetree)
            newtree.close()

            os.system('%s --alisim %s/temp_simulation -t %s/temp_tree.treefile -m %s --length %r --redo  >/dev/null 2>&1' % (self.IQ_Tree, self.Temp, self.Temp, self.Model, self.Length))

            # get the alignment
            align, sporder = utils.readPhylip('%s/temp_simulation.phy' % self.Temp)

            # remove temporary files
            os.system('rm -r %s' % self.Temp)

            # calculate proportion invariant
            pinv = self.countPinvIQTree(align)
            all_sites.append(pinv)

        return all_sites

    def simulateLambdaTree(self, birthRate):
        """Simulate data in IQTree under some lambda and a random tree topology."""

        # do the simulation
        os.system('mkdir %s' % self.Temp)
        os.system('%s --alisim %s/temp_tree -t RANDOM{bd{%r/0}/%r} -m %s --length %r --redo >/dev/null 2>&1' % (self.IQ_Tree, self.Temp, birthRate, len(self.Input_Order), self.Model, self.Length))

        thetree = open('%s/temp_tree.treefile' % self.Temp, 'r').readlines()[0].strip()

        thetree = utils.add_bl(thetree, birthRate)
        thetree = utils.replace_names(tree_string = thetree, Input_Match_dict = self.Input_Match_dict)

        return(thetree)

    def simulateStartTree(self, birthRate):
        """Simulate data in IQTree under some lambda and a random tree topology."""

        # do the simulation
        os.system('mkdir %s' % self.Temp)
        os.system('%s --alisim %s/Random_start -t RANDOM{bd{%r/0}/%r} -m %s --length %r --redo >/dev/null 2>&1' % (self.IQ_Tree, self.Temp, birthRate, len(self.Input_Order), self.Model, 1))

        thetree = open('%s/Random_start.treefile' % self.Temp, 'r').readlines()[0].strip()

        thetree = utils.add_bl(thetree, birthRate)
        thetree = utils.replace_names(tree_string = thetree, Input_Match_dict = self.Input_Match_dict)
        os.system('rm -r %s' % self.Temp)

        return(thetree)


    def simulateonTree(self, treename, maxSNPs, birthRate):
        """Simulate data in IQTree under some lambda and a given tree topology."""

        # list to hold all results across reps
        all_generated_regions = []
        all_generated_pinv = []

        # do the simulation
        for rep in range(self.Chunks):
            
            #change BL on tree
            bl_tree = utils.add_bl(treename, birthRate)
            os.system('mkdir -p %s' % self.Temp)
            treefile = open('%s/temp_tree.treefile' % self.Temp, 'w')
            treefile.write(bl_tree)
            treefile.close()

            os.system('%s --alisim %s/temp_simulation -t %s/temp_tree.treefile -m %s --length %r --redo >/dev/null 2>&1' % (self.IQ_Tree, self.Temp, self.Temp, self.Model, self.Length))

            # get the alignment
            align, sporder = utils.readPhylip('%s/temp_simulation.phy' % self.Temp)

            # remove temporary files
            os.system('rm -r %s' % self.Temp)

            # convert to np array
            sim_pinv, sim_matrix = self.sequencetonparray(align, sporder, maxSNPs)

            # add to full array
            all_generated_regions.append(sim_matrix)
            all_generated_pinv.append(sim_pinv)

        all_generated_regions = np.array(all_generated_regions)
        all_generated_pinv = np.array(all_generated_pinv)


        return(all_generated_regions, all_generated_pinv)


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

    def countPinvChunk(self):
        """Calculate the proportion of invariant sites from a chunk of the empirical data."""
        maxlength = self.Input_Alignment.get_alignment_length()

        all_sites = []
        for i in range(0, self.Chunks):
            alignmentstart = random.randrange(0, maxlength - self.Length)
            countvarsites = 0
            for i in range(alignmentstart, alignmentstart+self.Length):
                sequence = list(self.Input_Alignment[:, i])
                if len(set((sequence))) > 1:
                    countvarsites+=1
            # calculate proportion invariant sites
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
