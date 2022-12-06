import dendropy
import numpy as np
import re
from Bio import AlignIO
from matplotlib import pyplot as plt
import os
import random
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from io import StringIO
from Bio import Phylo
import ete3
import sys
import simulators
import itertools

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob,
    along with the probability of NNI vs TBR vs SPR."""
    return 1 - i/(num_iter-1) # start at 1, end at 0

def calc_max_snps(empirical_alignments, Length, Chunks):
    """Calculate the number of SNPs in samples from the empirical data."""

    all_sites = []
    for j in range(Chunks):
        # sample the empirical data
        current_length = 0
        sampled = []
        while current_length < Length:
            new_length_sampled = random.sample(range(len(empirical_alignments)),1)[0]
            sampled.append(new_length_sampled)
            new_length = empirical_alignments[new_length_sampled].max_sequence_size
            current_length += new_length

        countvarsites = 0
        current_length = 0
        for i in range(len(sampled)):
            countvarsites = 0
            charmatrix = empirical_alignments[sampled[i]]
            current_length += charmatrix.max_sequence_size
            if current_length <= Length:
                countvarsites += dendropy.calculate.popgenstat.num_segregating_sites(charmatrix, ignore_uncertain = True)
            else:
                to_sample = charmatrix.max_sequence_size - (current_length - Length)
                extracted = charmatrix.export_character_indices([*range(to_sample)])
                countvarsites += dendropy.calculate.popgenstat.num_segregating_sites(extracted, ignore_uncertain = True)
        all_sites.append(countvarsites)
    average_sites = int(np.ceil(np.average(all_sites)))
    return(average_sites)


def add_bl_coal(t, coal_time):
    prev_age = 0
    try:
        for nd in t.postorder_node_iter():
            if nd.is_leaf():
                nd.age = 0
            else:
                this_age = np.random.exponential(scale = coal_time)
                nd.age = this_age + prev_age
                prev_age = nd.age
        t.set_edge_lengths_from_node_ages()
    except:
        t = dendropy.Tree.get(data=t, schema="newick")
        for nd in t.postorder_node_iter():
            if nd.is_leaf():
                nd.age = 0
            else:
                this_age = np.random.exponential(scale = coal_time)
                nd.age = this_age + prev_age
                prev_age = nd.age
        t.set_edge_lengths_from_node_ages()
    return(t)

def add_bl(tree, birthRate):
    """Add branch lengths to trees."""
    thetree = dendropy.Tree.get(data=tree, schema="newick")
    for edge in thetree.preorder_edge_iter():
        new_length = np.random.exponential(scale = 1/ birthRate)
        edge.length = new_length
    thetree.reroot_at_midpoint()
    for edge in thetree.preorder_edge_iter():
        new_length = np.random.exponential(scale = 1/ birthRate)
        edge.length = new_length
    thetree.deroot()
    return(thetree.as_string(schema="newick", suppress_rooting = True))

def replace_names(tree_string, Input_Match_dict):
    for key in Input_Match_dict.keys():
        tree_string = re.sub(key+':', Input_Match_dict[key]+'_renamed:', tree_string)
    for key in Input_Match_dict.keys():
        tree_string = re.sub(Input_Match_dict[key]+'_renamed:', Input_Match_dict[key]+':', tree_string)

    return(tree_string)

def read_params_file(paramfilename):
    """This function will read parameters from the parameter input file."""

    paraminfo = open(paramfilename, 'r').readlines()
    try:
        IQTree_path = [x for x in paraminfo if 'IQTree path' in x][0].split(' = ')[1].split("#")[0].strip()
        alignment_folder = [x for x in paraminfo if 'alignment folder' in x][0].split(' = ')[1].split("#")[0].strip()
        Nucleotide_substitution_model = [x for x in paraminfo if 'Nucleotide substitution model' in x][0].split(' = ')[1].split("#")[0].strip()
        Min_scale = [x for x in paraminfo if 'Min scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_scale = [x for x in paraminfo if 'Max scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Min_coal = [x for x in paraminfo if 'Min coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_coal = [x for x in paraminfo if 'Max coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Width_scale = [x for x in paraminfo if 'Width scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Width_coal = [x for x in paraminfo if 'Width coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Length = [x for x in paraminfo if 'Length = ' in x][0].split(' = ')[1].split("#")[0].strip()
        Chunks = [x for x in paraminfo if 'Chunks' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_1_Iterations = ([x for x in paraminfo if 'Stage 1 Iterations' in x][0].split(' = ')[1].split("#")[0].strip())
        Stage_1_Proposals = [x for x in paraminfo if 'Stage 1 Proposals' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Pretraining_Iterations = [x for x in paraminfo if 'Stage 2 Pre-training Iterations' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Training_Iterations = [x for x in paraminfo if 'Stage 2 Training Iterations' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Pretraining_Epochs = [x for x in paraminfo if 'Stage 2 Pre-training Epochs' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Training_Epochs = [x for x in paraminfo if 'Stage 2 Training Epochs' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_trees = [x for x in paraminfo if 'Max Trees' in x][0].split(' = ')[1].split("#")[0].strip()
        true_tree = [x for x in paraminfo if 'true tree' in x][0].split(' = ')[1].split("#")[0].strip()
        results = [x for x in paraminfo if 'results' in x][0].split(' = ')[1].split("#")[0].strip()
        temp = [x for x in paraminfo if 'temp' in x][0].split(' = ')[1].split("#")[0].strip()
        start_tree = [x for x in paraminfo if 'start tree' in x][0].split(' = ')[1].split("#")[0].strip()
        checkpoint_interval = [x for x in paraminfo if 'checkpoint interval' in x][0].split(' = ')[1].split("#")[0].strip()
    except:
        sys.exit('ERROR: Some parameters are not defined in the params file. To use defaults, set a parameter to "None"; do not remove it entirely.')

    if Width_scale == 'None':
        Width_scale = (float(Max_scale) + float(Min_scale))/2/2
    if Width_coal == 'None':
        Width_coal = (float(Max_coal) + float(Min_coal))/2/2
    if Stage_1_Proposals == 'None':
        Stage_1_Proposals = 15
    if Stage_2_Training_Epochs == 'None':
        Stage_2_Training_Epochs == Stage_2_Pretraining_Epochs
    if Max_trees == 'None':
        Max_trees = 5
    allparams = [IQTree_path, alignment_folder, Nucleotide_substitution_model, Min_scale, Max_scale, Min_coal, Max_coal,
        Width_scale, Width_coal, Length, Chunks, Stage_1_Iterations, Stage_1_Proposals, Stage_2_Pretraining_Iterations,
        Stage_2_Training_Iterations, Stage_2_Pretraining_Epochs, Stage_2_Training_Epochs, Max_trees,results, temp, checkpoint_interval]
    missing_info = any(x=='None' for x in allparams)
    if missing_info:
        sys.exit('ERROR: Some required parameters were not supplied. Please check the params file.')
    else:
        return({'IQTree_path': IQTree_path, 'alignment_folder': alignment_folder, 'Nucleotide_substitution_model': Nucleotide_substitution_model, 'Min_scale': float(Min_scale), 'Max_scale': float(Max_scale), 'Min_coal': float(Min_coal), 'Max_coal': float(Max_coal),
            'Width_scale': float(Width_scale), 'Width_coal': float(Width_coal), 'Length': int(Length), 'Chunks': int(Chunks), 'Stage_1_Iterations': int(Stage_1_Iterations), 'Stage_1_Proposals': int(Stage_1_Proposals), 'Stage_2_Pretraining_Iterations': int(Stage_2_Pretraining_Iterations),
            'Stage_2_Training_Iterations': int(Stage_2_Training_Iterations), 'Stage_2_Pretraining_Epochs': int(Stage_2_Pretraining_Epochs), 'Stage_2_Training_Epochs': int(Stage_2_Training_Epochs), 'Max_trees': int(Max_trees), 'true_tree': true_tree,
            'results': results, 'temp': temp, 'start_tree': start_tree, 'checkpoint_interval': checkpoint_interval})

def create_matchdict(sporder):
    """This will take the empirical sporder and create a dictionary mapping to default alisim names (T1-T#taxa). The returned dict will be used to rename alisim trees."""
    matchdict = {}
    for item in range(1, len(sporder)+1):
        string = 'T' + str(item)
        matchdict[string] = sporder[item-1]
    return(matchdict)

def write_results_stage1(output, scales, coal, distances):
    try:
        os.mkdir(output)
    except OSError:
        pass
    xaxis = list(range(1, len(scales)+1))
    xaxis = [str(x) for x in xaxis]
    plt.plot(xaxis, scales, label ="Scale")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/Scales.png')
    plt.close()
    scalefile = open(output+'/Scales.txt', 'w')
    for thevalue in scales:
        scalefile.write(str(thevalue)+'\n')
    scalefile.close()
    
    plt.plot(xaxis, coal, label ="Coal")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/Coal.png')
    plt.close()
    coalfile = open(output+'/Coal.txt', 'w')
    for thevalue in coal:
        coalfile.write(str(thevalue)+'\n')
    coalfile.close()
    
    plt.plot(xaxis, distances, label ="Distance")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/Distances.png')
    plt.close()
    distancefile = open(output+'/Distances.txt', 'w')
    for thevalue in distances:
        distancefile.write(str(thevalue)+'\n')
    distancefile.close()

def write_results_stage2(trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, output, true_tree):
    # create the x axis
    xaxis = list(range(1, len(generator_loss)+1))
    xaxis = [str(x) for x in xaxis]

    # write the trees
    treefile = open(output+'/Trees.txt', 'w')
    for tree in trees:
        treefile.write(tree.strip()+'\n')
    treefile.close()

    # plot loss
    plt.plot(xaxis, generator_loss, label ="Generator Loss")
    plt.plot(xaxis, discriminator_loss, label ="Discriminator Loss")
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(fname=output+'/Loss.png')
    plt.close()

    # write generator loss
    generatorlossfile = open(output+'/GeneratorLoss.txt', 'w')
    for thevalue in generator_loss:
        generatorlossfile.write(str(thevalue)+'\n')
    generatorlossfile.close()

    # write discriminator loss
    discriminatorlossfile = open(output+'/DiscriminatorLoss.txt', 'w')
    for thevalue in discriminator_loss:
        discriminatorlossfile.write(str(thevalue)+'\n')
    discriminatorlossfile.close()

    # plot and write generator accuracy
    plt.plot(xaxis, generator_fake_acc, label ="Generator Fake Acc")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/GeneratorFakeAcc.png')
    plt.close()
    generatorfakeaccfile = open(output+'/GeneratorFakeAcc.txt', 'w')
    for thevalue in generator_fake_acc:
        generatorfakeaccfile.write(str(thevalue)+'\n')
    generatorfakeaccfile.close()

    # plot disc fake and real accuracy
    plt.plot(xaxis, discriminator_fake_acc, label ="Discriminator Fake Acc")
    plt.plot(xaxis, discriminator_real_acc, label ="Discriminator Real Acc")
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(fname=output+'/DiscriminatorAcc.png')
    plt.close()

    # write disc fake acc
    discriminatorfakeaccfile = open(output+'/DiscriminatorFakeAcc.txt', 'w')
    for thevalue in discriminator_fake_acc:
        discriminatorfakeaccfile.write(str(thevalue)+'\n')
    discriminatorfakeaccfile.close()

    # write disc real acc
    discriminatorrealaccfile = open(output+'/DiscriminatorRealAcc.txt', 'w')
    for thevalue in discriminator_real_acc:
        discriminatorrealaccfile.write(str(thevalue)+'\n')
    discriminatorrealaccfile.close()

    # if it's a simulation, plot and write RF
    if true_tree!= 'None':
        ref_tree = ete3.Tree(true_tree)
        rf_distances = []
        for tree in trees:
            comp_tree = ete3.Tree(tree)
            comp = comp_tree.compare(ref_tree, unrooted=True)
            rf_distances.append(comp['norm_rf'])
        # plot and write rf distances
        plt.plot(xaxis, rf_distances, label ="RF distances Acc")
        plt.xlabel('Iteration')
        plt.ylim(top=1)
        plt.ylim(bottom=0)
        plt.savefig(fname=output+'/RF_Distances.png')
        plt.close()
        rfdistancesfile = open(output+'/RFdistances.txt', 'w')
        for thevalue in rf_distances:
            rfdistancesfile.write(str(thevalue)+'\n')
        rfdistancesfile.close()

def write_results_stage2_generator(trees, distances, output):
    # create the x axis
    xaxis = list(range(1, len(distances)+1))
    xaxis = [str(x) for x in xaxis]

    # write the trees
    treefile = open(output+'/Trees.txt', 'w')
    for tree in trees:
        treefile.write(tree+'\n')
    treefile.close()

    # plot distance
    plt.plot(xaxis, distances, label ="Distances")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/Distances.png')
    plt.close()

    # write generator loss
    distancefile = open(output+'/Distances.txt', 'w')
    for thevalue in distances:
        distancefile.write(str(thevalue)+'\n')
    distancefile.close()



def sequencetonparrayChunk(empirical_alignments, numTaxa, maxSNPs, Length, Chunks, sampled, Input_Order, Input_Match_dict):
    """Convert a fasta chunk to a numpy array for downstream operations."""
    all_empirical_regions = []
    all_empirical_pinv = []
    for i in range(len(sampled)):
        current_length = 0
        for j in range(len(sampled[i])):
            charmatrix = empirical_alignments[sampled[i][j]]
            current_length += charmatrix.max_sequence_size
            if current_length > Length:
                to_sample = charmatrix.max_sequence_size - (current_length - Length)
                extracted = charmatrix.export_character_indices([*range(to_sample)])
                charmatrix=extracted
            if j == 0:
                this_charmatrix = charmatrix
            else:
                this_charmatrix = dendropy.DnaCharacterMatrix.concatenate([this_charmatrix, charmatrix])

        tempsim = simulators.Simulator(IQ_Tree=None, Model=None, Chunks=Chunks, Length=Length, Input_Alignments=empirical_alignments, Input_Lengths=None, Input_Order=Input_Order, Input_Match_dict=Input_Match_dict, Temp=None)
        this_prop_inv, this_matrix_full = tempsim.sequencetonparray_dendropy(this_charmatrix, Input_Order, maxSNPs, True)
        # add to full array
        all_empirical_regions.append(this_matrix_full)
        all_empirical_pinv.append(this_prop_inv)

    all_empirical_regions = np.array(all_empirical_regions)
    all_empirical_pinv = np.array(all_empirical_pinv)

    return(all_empirical_regions, all_empirical_pinv)


def get_start_tree(start_tree, coal, Input_Alignments, Results, Input_Order, Input_Match_dict):

    if start_tree == 'Random':
        print('Generating a random start tree.')
        random_simulator = simulators.Simulator(None, None, None, None, Input_Alignments, None, Input_Order, Input_Match_dict, None)
        start_tree = random_simulator.simulateStartTree(coal)
        return(start_tree)

    else:
        try:
            ete3.Tree(start_tree)
            return(start_tree)
        except:
            sys.exit('ERROR: Check starting tree argument in params file.')

def simulatePseudo(iqTree, birthRate, model, numTaxa, length, output):
    """Simulate data in IQTree under some lambda and a random tree topology."""
    print(output)
    # do the simulation
    os.system('%s --alisim %s -t RANDOM{bd{%r/0}/%r} -m %s --length %r --redo >/dev/null 2>&1 --redo' % (iqTree, output, birthRate, numTaxa, model, length))

    thetree = open('%s.treefile' % output, 'r').readlines()[0].strip()

    thetree = add_bl(thetree, birthRate)

    newtree = open('%s.treefile' % output, 'w')
    newtree.write(thetree)
    newtree.close()

    os.system('%s --alisim %s -t %s.treefile -m %s --length %r --redo >/dev/null 2>&1' % (iqTree, output, output, model, length))
    #print('IS THE PSEUDOOBSERVED DATA THERE?')


def create_checkpoint(trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, output, MaxSNPs):

    # create the x axis
    xaxis = list(range(1, len(generator_loss)+1))
    xaxis = [str(x) for x in xaxis]

    # write the trees
    treefile = open(output+'/Checkpoint_trees.txt', 'w')
    for tree in trees:
        try:
            treefile.write(tree.strip())
        except:
            treefile.write(tree.as_string(schema="newick"))
        treefile.write('\n')
    treefile.close()


    # write generator loss
    generatorlossfile = open(output+'/Checkpoint_GeneratorLoss.txt', 'w')
    for thevalue in generator_loss:
        generatorlossfile.write(str(thevalue)+'\n')
    generatorlossfile.close()

    # write discriminator loss
    discriminatorlossfile = open(output+'/Checkpoint_DiscriminatorLoss.txt', 'w')
    for thevalue in discriminator_loss:
        discriminatorlossfile.write(str(thevalue)+'\n')
    discriminatorlossfile.close()

    # write generator accuracy
    generatorfakeaccfile = open(output+'/Checkpoint_GeneratorFakeAcc.txt', 'w')
    for thevalue in generator_fake_acc:
        generatorfakeaccfile.write(str(thevalue)+'\n')
    generatorfakeaccfile.close()

    # write disc fake acc
    discriminatorfakeaccfile = open(output+'/Checkpoint_DiscriminatorFakeAcc.txt', 'w')
    for thevalue in discriminator_fake_acc:
        discriminatorfakeaccfile.write(str(thevalue)+'\n')
    discriminatorfakeaccfile.close()

    # write disc real acc
    discriminatorrealaccfile = open(output+'/Checkpoint_DiscriminatorRealAcc.txt', 'w')
    for thevalue in discriminator_real_acc:
        discriminatorrealaccfile.write(str(thevalue)+'\n')
    discriminatorrealaccfile.close()

    # write max SNPs
    maxsnpfile = open(output+'/Checkpoint_MaxSNPs.txt', 'w')
    maxsnpfile.write(str(MaxSNPs)+'\n')
    maxsnpfile.close()

class Parse_Empirical(object):

    def __init__(self, alignment_folder):
        self.alignment_folder = alignment_folder


    def __call__(self):
        empirical_alignments = os.listdir(self.alignment_folder)
        empirical_alignments = [x for x in empirical_alignments if x.endswith('.phy')]
        empirical_charmatrices = []
        empirical_pinv = []
        empirical_lengths = []
        first_char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/%s' % (self.alignment_folder, empirical_alignments[0])), schema = "phylip")
        namespace = first_char_matrix.taxon_namespace
        taxon_order = first_char_matrix.taxon_namespace
        taxon_order = [str(x).strip("'") for x in taxon_order]
        num_taxa = len(first_char_matrix)
        del first_char_matrix
        for alignment in empirical_alignments:
            char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/%s' % (self.alignment_folder, alignment)), schema = "phylip", taxon_namespace = namespace)
            empirical_charmatrices.append(char_matrix)
            countvarsites = dendropy.calculate.popgenstat.num_segregating_sites(char_matrix, ignore_uncertain=True)
            prop_inv = [1- (countvarsites / char_matrix.max_sequence_size)]
            empirical_pinv.append(prop_inv)
            empirical_lengths.append(char_matrix.max_sequence_size)

        return(empirical_charmatrices, empirical_pinv, empirical_lengths, num_taxa, taxon_order)

