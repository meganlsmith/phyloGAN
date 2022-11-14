import numpy as np
import dendropy
import os
from matplotlib import pyplot as plt
import ete3
import sys
import itertools
import re

def replace_names(tree_string, Input_Match_dict):
    try:
        tree_string = tree_string.as_string(schema="newick", suppress_rooting = True)
    except:
        pass 
    for key in Input_Match_dict.keys():
        to_replace = key+":"
        replacement = str(Input_Match_dict[key]).strip("'")+'_renamed:'
        replacement2 = str(Input_Match_dict[key]).strip("'")+':'

        tree_string = re.sub(to_replace, replacement, tree_string)

    for key in Input_Match_dict.keys():
        replacement = str(Input_Match_dict[key]).strip("'")+'_renamed:'
        replacement2 = str(Input_Match_dict[key]).strip("'")+':'
        tree_string = re.sub(replacement, replacement2, tree_string)

    if isinstance(tree_string, str):
        tree_string = dendropy.Tree.get(data=tree_string, schema="newick")
    return(tree_string)

def create_matchdict(sporder):
    """This will take the empirical sporder and create a dictionary mapping to default alisim names (T1-T#taxa). The returned dict will be used to rename alisim trees."""
    matchdict = {}
    for item in range(1, len(sporder)+1):
        string = 'T' + str(item)
        matchdict[string] = sporder[item-1]
    return(matchdict)


def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob,
    along with the probability of NNI vs TBR vs SPR."""
    return 1 - i/(num_iter-1) # start at 1, end at 0

def add_bl(t, coal_time):
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

def write_results_stage1(coal, scale, output):

    #  make directory
    os.system('mkdir -p %s' % output)

    # write generator loss
    coaltimefile = open(output+'/CoalTime.txt', 'w')
    for thevalue in coal:
        coaltimefile.write(str(thevalue)+'\n')
    coaltimefile.close()

    # write discriminator loss
    scalefile = open(output+'/Scale.txt', 'w')
    for thevalue in scale:
        scalefile.write(str(thevalue)+'\n')
    scalefile.close()



def write_results_stage2(trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, output, true_tree):

    #  make directory
    os.system('mkdir -p %s' % output)

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

def read_params_file(paramfilename):
    """This function will read parameters from the parameter input file."""

    paraminfo = open(paramfilename, 'r').readlines()
    try:
        IQTree_path = [x for x in paraminfo if 'IQTree path' in x][0].split(' = ')[1].split("#")[0].strip()
        alignment_folder = [x for x in paraminfo if 'alignment folder' in x][0].split(' = ')[1].split("#")[0].strip()
        Nucleotide_substitution_model = [x for x in paraminfo if 'Nucleotide substitution model' in x][0].split(' = ')[1].split("#")[0].strip()
        Min_coal = [x for x in paraminfo if 'Min coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_coal = [x for x in paraminfo if 'Max coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Width_coal = [x for x in paraminfo if 'Width coal' in x][0].split(' = ')[1].split("#")[0].strip()
        Min_scale = [x for x in paraminfo if 'Min scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_scale = [x for x in paraminfo if 'Max scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Width_scale = [x for x in paraminfo if 'Width scale' in x][0].split(' = ')[1].split("#")[0].strip()
        Gene_tree_samples = [x for x in paraminfo if 'Gene tree samples' in x][0].split(' = ')[1].split("#")[0].strip()
        Length = [x for x in paraminfo if 'Length' in x][0].split(' = ')[1].split("#")[0].strip()
        Batch_size = [x for x in paraminfo if 'Batch size' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_1_Iterations = ([x for x in paraminfo if 'Stage 1 Iterations' in x][0].split(' = ')[1].split("#")[0].strip())
        Stage_1_Proposals = [x for x in paraminfo if 'Stage 1 Proposals' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Pretraining_Iterations = [x for x in paraminfo if 'Stage 2 Pre-training Iterations' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Training_Iterations = [x for x in paraminfo if 'Stage 2 Training Iterations' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Pretraining_Epochs = [x for x in paraminfo if 'Stage 2 Pre-training Epochs' in x][0].split(' = ')[1].split("#")[0].strip()
        Stage_2_Training_Epochs = [x for x in paraminfo if 'Stage 2 Training Epochs' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_trees = [x for x in paraminfo if 'Max Trees' in x][0].split(' = ')[1].split("#")[0].strip()
        Simulate_pseudoobserved = [x for x in paraminfo if 'Simulate_pseudoobserved' in x][0].split(' = ')[1].split("#")[0].strip()
        true_tree = [x for x in paraminfo if 'true tree' in x][0].split(' = ')[1].split("#")[0].strip()
        results = [x for x in paraminfo if 'results' in x][0].split(' = ')[1].split("#")[0].strip()
        temp = [x for x in paraminfo if 'temp' in x][0].split(' = ')[1].split("#")[0].strip()
        start_tree = [x for x in paraminfo if 'start tree' in x][0].split(' = ')[1].split("#")[0].strip()
        check_point = [x for x in paraminfo if 'check point' in x][0].split(' = ')[1].split("#")[0].strip()
    except:
        sys.exit('ERROR: Some parameters are not defined in the params file. To use defaults, set a parameter to "None"; do not remove it entirely.')

    if Simulate_pseudoobserved == 'True':
        try:
            Simulation_nucleotide_substitution_model = [x for x in paraminfo if 'Simulation nucleotide substitution model' in x][0].split(' = ')[1].split("#")[0].strip()
            Gene_trees = [x for x in paraminfo if 'Gene trees' in x][0].split(' = ')[1].split("#")[0].strip()
            Num_taxa = [x for x in paraminfo if 'Num taxa' in x][0].split(' = ')[1].split("#")[0].strip()
            Coal_time = [x for x in paraminfo if 'Coal time' in x][0].split(' = ')[1].split("#")[0].strip()
            Scale = [x for x in paraminfo if 'Scale ' in x][0].split(" = ")[1].split("#")[0].strip()
            Output_pseudoobserved = [x for x in paraminfo if 'Output pseudoobserved' in x][0].split(' = ')[1].split("#")[0].strip()
            true_tree = Output_pseudoobserved + '/species_tree.tre'
        except:
            sys.exit('ERROR: Some parameters are not defined in the params file. To use defaults, set a parameter to "None"; do not remove it entirely.')


    if Width_coal == 'None':
        Width_coal = (float(Max_coal) + float(Min_coal))/2/2
    if Width_scale == 'None':
        Width_scale = (float(Max_scale) + float(Min_scale))/2/2
    if Stage_1_Proposals == 'None':
        Stage_1_Proposals = 15
    if Stage_2_Training_Epochs == 'None':
        Stage_2_Training_Epochs == Stage_2_Pretraining_Epochs
    if Max_trees == 'None':
        Max_trees = 5
    if Simulate_pseudoobserved == 'True' and true_tree == 'None':
        true_tree = Output_pseudoobserved + '.treefile'

    if Simulate_pseudoobserved == 'False':
        allparams = [IQTree_path, alignment_folder, Nucleotide_substitution_model, Min_coal, Max_coal,
            Width_coal, Min_scale, Max_scale, Width_scale, Gene_tree_samples, Length, Batch_size,
            Stage_1_Iterations, Stage_1_Proposals, Stage_2_Pretraining_Iterations,
            Stage_2_Training_Iterations, Stage_2_Pretraining_Epochs, Stage_2_Training_Epochs, Max_trees,results, temp,
            start_tree, check_point]
        missing_info = any(x=='None' for x in allparams)
        if missing_info:
            sys.exit('ERROR: Some required parameters were not supplied. Please check the params file.')
        else:
            return({'IQTree_path': IQTree_path, 'alignment_folder': alignment_folder, 'Nucleotide_substitution_model': Nucleotide_substitution_model, 'Min_coal': float(Min_coal), 'Max_coal': float(Max_coal),
                'Width_coal': float(Width_coal), 'Min_scale': float(Min_scale), 'Max_scale': float(Max_scale), 'Width_scale': float(Width_scale), 'Gene_tree_samples': int(Gene_tree_samples),
                'Length': int(Length), 'Batch_size': int(Batch_size), 'Stage_1_Iterations': int(Stage_1_Iterations), 'Stage_1_Proposals': int(Stage_1_Proposals), 'Stage_2_Pretraining_Iterations': int(Stage_2_Pretraining_Iterations),
                'Stage_2_Training_Iterations': int(Stage_2_Training_Iterations), 'Stage_2_Pretraining_Epochs': int(Stage_2_Pretraining_Epochs), 'Stage_2_Training_Epochs': int(Stage_2_Training_Epochs), 'Max_trees': int(Max_trees), 'Simulate_pseudoobserved': False,'true_tree': true_tree,
                'results': results, 'temp': temp, 'start_tree': start_tree, 'check_point': check_point})

    else:
        allparams = [IQTree_path, Nucleotide_substitution_model, Min_coal, Max_coal,
            Width_coal, Min_scale, Max_scale, Width_scale, Gene_tree_samples, Length, Batch_size,
            Stage_1_Iterations, Stage_1_Proposals, Stage_2_Pretraining_Iterations,
            Stage_2_Training_Iterations, Stage_2_Pretraining_Epochs, Stage_2_Training_Epochs, Max_trees,results, temp,
            start_tree,
            Simulation_nucleotide_substitution_model, Gene_trees, Num_taxa, Coal_time, Scale, Output_pseudoobserved, true_tree, check_point]
        missing_info = any(x=='None' for x in allparams)
        if missing_info:
            sys.exit('ERROR: Some required parameters were not supplied. Please check the params file.')
        else:
            return({'IQTree_path': IQTree_path, 'alignment_folder': Output_pseudoobserved + '/alignments/', 'Nucleotide_substitution_model': Nucleotide_substitution_model, 'Min_coal': float(Min_coal), 'Max_coal': float(Max_coal),
                'Width_coal': float(Width_coal), 'Min_scale': float(Min_scale), 'Max_scale': float(Max_scale), 'Width_scale': float(Width_scale), 'Gene_tree_samples': int(Gene_tree_samples),
                'Length': int(Length), 'Batch_size': int(Batch_size), 'Stage_1_Iterations': int(Stage_1_Iterations), 'Stage_1_Proposals': int(Stage_1_Proposals), 'Stage_2_Pretraining_Iterations': int(Stage_2_Pretraining_Iterations),
                'Stage_2_Training_Iterations': int(Stage_2_Training_Iterations), 'Stage_2_Pretraining_Epochs': int(Stage_2_Pretraining_Epochs), 'Stage_2_Training_Epochs': int(Stage_2_Training_Epochs), 'Max_trees': int(Max_trees), 'Simulate_pseudoobserved': True, 'true_tree': true_tree,
                'results': results, 'temp': temp, 'start_tree': start_tree,
                'Simulation_nucleotide_substitution_model': Simulation_nucleotide_substitution_model, 'Gene_trees': int(Gene_trees), 'Num_taxa': int(Num_taxa), 
                'Coal_time': float(Coal_time), 'Scale': float(Scale), 'Output_pseudoobserved': Output_pseudoobserved, 'true_tree': true_tree, 'check_point': check_point})

def count_pis(char_matrix, num_taxa):
    sequence_list = []
    for i in range(num_taxa):
        sequence_list.append(list(str(char_matrix.sequences()[i])))
    pis = 0
    for bp in range(len(sequence_list[0])):
        letters = []
        for taxon in range(num_taxa):
            letters.append(sequence_list[taxon][bp])
        letter_counts = [letters.count(x) for x in ['A', 'T', 'C', 'G']]
        mult_counts = [x for x in letter_counts if x >= 2]
        if len(mult_counts) > 1:
            pis+=1
    return(pis)

def count_var(char_matrix, num_taxa):
    sequence_list = []
    for i in range(num_taxa):
        sequence_list.append(list(str(char_matrix.sequences()[i])))
    var_sites = 0
    for bp in range(len(sequence_list[0])):
        letters = []
        for taxon in range(num_taxa):
            letters.append(sequence_list[taxon][bp])
        if len(set(letters)) > 1:
            var_sites+=1
    return(var_sites)

def charmatrix_to_nparray(char_matrix_list, max_var, num_taxa, taxon_order):

    variable_sites = []    
    full_char_matrix = np.empty(shape=(0, num_taxa))
    for charmatrix in char_matrix_list:
        this_character_matrix = []
        for taxon in taxon_order:
            try:
                taxon_sequence = list(str((charmatrix['%s' % (str(taxon).strip("'"))])))
            except:        
                taxon_sequence = list(str((charmatrix['%s_%s' % (str(taxon).strip("'"), '1')])))

            # convert to strings of A=0, T=1, C=2, G=3
            taxon_sequence = [0 if x == 'A' else x for x in taxon_sequence]
            taxon_sequence = [1 if x == 'T' else x for x in taxon_sequence]
            taxon_sequence = [2 if x == 'C' else x for x in taxon_sequence]
            taxon_sequence = [3 if x == 'G' else x for x in taxon_sequence]
            this_character_matrix.append(taxon_sequence)

        this_character_array = np.array(this_character_matrix)
        variable_character_array=np.empty(shape=(0, num_taxa))
        count_var = 0
        
        for site in range(this_character_array.shape[1]):
            site_values = this_character_array[:,site]
            if len(set(site_values)) > 1 and count_var < max_var:
                variable_character_array = np.append(variable_character_array, np.array([site_values]), axis=0)
                count_var+=1
            elif len(set(site_values)) > 1:
                count_var+=1

        # pad with 4s as needed
        while variable_character_array.shape[0] < max_var:
            temparray=[4]*num_taxa
            variable_character_array = np.append(variable_character_array, np.array([temparray]), axis=0)
        full_char_matrix = np.append(full_char_matrix, variable_character_array, axis=0)
        variable_sites.append(count_var / this_character_array.shape[1])
    # transpose genotype matrix so individuals: rows, SNPS: columns
    full_char_matrix = np.transpose(full_char_matrix)

    #sample all combos of four
    combos = list(itertools.combinations(range(full_char_matrix.shape[0]), 4))
    combo_rows = []
    for i in combos:
        combo_rows.append(full_char_matrix[i,:])
    matrix_full = np.concatenate(combo_rows)
    
    
    return(matrix_full, variable_sites)


class Parse_Empirical(object):

    def __init__(self, alignment_folder):
        self.alignment_folder = alignment_folder


    def __call__(self):
        empirical_alignments = os.listdir(self.alignment_folder)
        empirical_alignments = [x for x in empirical_alignments if x.endswith('.phy')]
        empirical_charmatrices = []
        empirical_pis = []
        empirical_lengths = []
        empirical_var = []
        first_char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/%s' % (self.alignment_folder, empirical_alignments[0])), schema = "phylip")
        taxon_order = first_char_matrix.taxon_namespace
        num_taxa = len(first_char_matrix)
        del first_char_matrix
        for alignment in empirical_alignments:
            char_matrix = dendropy.DnaCharacterMatrix.get(file=open('%s/%s' % (self.alignment_folder, alignment)), schema = "phylip")
            empirical_charmatrices.append(char_matrix)
            empirical_pis.append(count_pis(char_matrix, num_taxa))
            empirical_var.append(count_var(char_matrix, num_taxa))
            empirical_lengths.append(char_matrix.max_sequence_size)
    
        return(empirical_charmatrices, empirical_pis, empirical_lengths, empirical_var, num_taxa, taxon_order)

def create_checkpoint(trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, output, MaxSNPs):

    # create the x axis
    xaxis = list(range(1, len(generator_loss)+1))
    xaxis = [str(x) for x in xaxis]

    # write the trees
    treefile = open(output+'/Checkpoint_trees.txt', 'w')
    for tree in trees:
        treefile.write(tree.strip())
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

