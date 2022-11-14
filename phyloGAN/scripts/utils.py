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

def readPhylip(phylipHandle):
    """Read the phylip file."""
    align = AlignIO.read(phylipHandle, format="phylip")
    sporder = []
    for record in align:
        sporder.append(record.id)
    return align,sporder

def read_params_file(paramfilename):
    """This function will read parameters from the parameter input file."""

    paraminfo = open(paramfilename, 'r').readlines()
    try:
        IQTree_path = [x for x in paraminfo if 'IQTree path' in x][0].split(' = ')[1].split("#")[0].strip()
        alignment_file = [x for x in paraminfo if 'alignment file' in x][0].split(' = ')[1].split("#")[0].strip()
        Nucleotide_substitution_model = [x for x in paraminfo if 'Nucleotide substitution model' in x][0].split(' = ')[1].split("#")[0].strip()
        Min_lambda = [x for x in paraminfo if 'Min lambda' in x][0].split(' = ')[1].split("#")[0].strip()
        Max_lambda = [x for x in paraminfo if 'Max lambda' in x][0].split(' = ')[1].split("#")[0].strip()
        Width_lambda = [x for x in paraminfo if 'Width lambda' in x][0].split(' = ')[1].split("#")[0].strip()
        Length = [x for x in paraminfo if 'Length' in x][0].split(' = ')[1].split("#")[0].strip()
        Chunks = [x for x in paraminfo if 'Chunks' in x][0].split(' = ')[1].split("#")[0].strip()
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
        checkpoint_interval = [x for x in paraminfo if 'checkpoint interval' in x][0].split(' = ')[1].split("#")[0].strip()
    except:
        sys.exit('ERROR: Some parameters are not defined in the params file. To use defaults, set a parameter to "None"; do not remove it entirely.')

    if Simulate_pseudoobserved == 'True':
        try:
            Simulation_nucleotide_substitution_model = [x for x in paraminfo if 'Simulation nucleotide substitution model' in x][0].split(' = ')[1].split("#")[0].strip()
            Full_length = [x for x in paraminfo if 'Full length' in x][0].split(' = ')[1].split("#")[0].strip()
            Num_taxa = [x for x in paraminfo if 'Num taxa' in x][0].split(' = ')[1].split("#")[0].strip()
            Birth_rate = [x for x in paraminfo if 'Birth rate' in x][0].split(' = ')[1].split("#")[0].strip()
            Output_pseudoobserved = [x for x in paraminfo if 'Output pseudoobserved' in x][0].split(' = ')[1].split("#")[0].strip()
        except:
            sys.exit('ERROR: Some parameters are not defined in the params file. To use defaults, set a parameter to "None"; do not remove it entirely.')


    if Width_lambda == 'None':
        Width_lambda = (float(Max_lambda) + float(Min_lambda))/2/2
    if Stage_1_Proposals == 'None':
        Stage_1_Proposals = 15
    if Stage_2_Training_Epochs == 'None':
        Stage_2_Training_Epochs == Stage_2_Pretraining_Epochs
    if Max_trees == 'None':
        Max_trees = 5
    if Simulate_pseudoobserved == 'True' and true_tree == 'None':
        true_tree = Output_pseudoobserved + '.treefile'

    if Simulate_pseudoobserved == 'False':
        allparams = [IQTree_path, alignment_file, Nucleotide_substitution_model, Min_lambda, Max_lambda,
            Width_lambda, Length, Chunks, Stage_1_Iterations, Stage_1_Proposals, Stage_2_Pretraining_Iterations,
            Stage_2_Training_Iterations, Stage_2_Pretraining_Epochs, Stage_2_Training_Epochs, Max_trees,results, temp, checkpoint_interval]
        missing_info = any(x=='None' for x in allparams)
        if missing_info:
            sys.exit('ERROR: Some required parameters were not supplied. Please check the params file.')
        else:
            return({'IQTree_path': IQTree_path, 'alignment_file': alignment_file, 'Nucleotide_substitution_model': Nucleotide_substitution_model, 'Min_lambda': float(Min_lambda), 'Max_lambda': float(Max_lambda),
                'Width_lambda': float(Width_lambda), 'Length': int(Length), 'Chunks': int(Chunks), 'Stage_1_Iterations': int(Stage_1_Iterations), 'Stage_1_Proposals': int(Stage_1_Proposals), 'Stage_2_Pretraining_Iterations': int(Stage_2_Pretraining_Iterations),
                'Stage_2_Training_Iterations': int(Stage_2_Training_Iterations), 'Stage_2_Pretraining_Epochs': int(Stage_2_Pretraining_Epochs), 'Stage_2_Training_Epochs': int(Stage_2_Training_Epochs), 'Max_trees': int(Max_trees), 'true_tree': true_tree,
                'results': results, 'temp': temp, 'start_tree': start_tree, 'checkpoint_interval': checkpoint_interval})

    else:
        allparams = [IQTree_path, Nucleotide_substitution_model, Min_lambda, Max_lambda,
            Width_lambda, Length, Chunks, Stage_1_Iterations, Stage_1_Proposals, Stage_2_Pretraining_Iterations,
            Stage_2_Training_Iterations, Stage_2_Pretraining_Epochs, Stage_2_Training_Epochs, Max_trees,results, temp,
            Simulation_nucleotide_substitution_model, Full_length, Num_taxa, Birth_rate, Output_pseudoobserved, true_tree, checkpoint_interval]
        missing_info = any(x=='None' for x in allparams)
        if missing_info:
            sys.exit('ERROR: Some required parameters were not supplied. Please check the params file.')
        else:
            return({'IQTree_path': IQTree_path, 'alignment_file': alignment_file, 'Nucleotide_substitution_model': Nucleotide_substitution_model, 'Min_lambda': float(Min_lambda), 'Max_lambda': float(Max_lambda),
                'Width_lambda': float(Width_lambda), 'Length': int(Length), 'Chunks': int(Chunks), 'Stage_1_Iterations': int(Stage_1_Iterations), 'Stage_1_Proposals': int(Stage_1_Proposals), 'Stage_2_Pretraining_Iterations': int(Stage_2_Pretraining_Iterations),
                'Stage_2_Training_Iterations': int(Stage_2_Training_Iterations), 'Stage_2_Pretraining_Epochs': int(Stage_2_Pretraining_Epochs), 'Stage_2_Training_Epochs': int(Stage_2_Training_Epochs), 'Max_trees': int(Max_trees), 'results': results, 'temp': temp, 'start_tree': start_tree,
                'Simulation_nucleotide_substitution_model': Simulation_nucleotide_substitution_model, 'Full_length': int(Full_length), 'Num_taxa': int(Num_taxa), 'Birth_rate': float(Birth_rate), 'Output_pseudoobserved': Output_pseudoobserved, 'true_tree': true_tree, 'checkpoint_interval': checkpoint_interval})

def create_matchdict(sporder):
    """This will take the empirical sporder and create a dictionary mapping to default alisim names (T1-T#taxa). The returned dict will be used to rename alisim trees."""
    matchdict = {}
    for item in range(1, len(sporder)+1):
        string = 'T' + str(item)
        matchdict[string] = sporder[item-1]
    return(matchdict)

def write_results_stage1(output, lambdas, distances):
    try:
        os.mkdir(output)
    except OSError:
        pass
    xaxis = list(range(1, len(lambdas)+1))
    xaxis = [str(x) for x in xaxis]
    plt.plot(xaxis, lambdas, label ="Lambda")
    plt.xlabel('Iteration')
    plt.savefig(fname=output+'/Lambdas.png')
    plt.close()
    lambdafile = open(output+'/Lambdas.txt', 'w')
    for thevalue in lambdas:
        lambdafile.write(str(thevalue)+'\n')
    lambdafile.close()

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



def calc_max_snps(align, chunklength, numAlignments):
    """Calculate the proportion of invariant sites from a chunk of the empirical data."""
    maxlength = align.get_alignment_length()

    all_sites = []
    for i in range(0, numAlignments):
        alignmentstart = random.randrange(0, maxlength - chunklength)
        countvarsites = 0
        for i in range(alignmentstart, alignmentstart+chunklength):
            sequence = list(align[:, i])
            if len(set((sequence))) > 1:
                countvarsites+=1
        # calculate proportion invariant sites
        all_sites.append(countvarsites)
    average_sites = int(np.ceil(np.average(all_sites)))
    return(average_sites)

def sequencetonparrayChunk(align, numTaxa, maxSNPs, Length, Chunks):
    """Convert a fasta chunk to a numpy array for downstream operations."""

    maxlength = align.get_alignment_length()

     # list to hold all results across reps
    all_empirical_regions = []
    all_empirical_pinv = []

    for i in range(Chunks):

        matrix_out=np.empty(shape=(0, numTaxa))
        countvarsites = 0

        alignmentstart = random.randrange(0, maxlength - Length)

        for i in range(alignmentstart, alignmentstart+Length):
            sequence = list(align[:, i])
            temparray=[]
            if len(set((sequence))) > 1 and countvarsites < maxSNPs:
                for individual in range(numTaxa):
                    if sequence[individual] == 'A':
                        temparray.append(0)
                    elif sequence[individual] == 'T':
                        temparray.append(1)
                    elif sequence[individual] == 'C':
                        temparray.append(2)
                    elif sequence[individual] == 'G':
                        temparray.append(3)
                    else:
                        temparray.append(4)
                countvarsites+=1
                matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)

            elif len(set((sequence))) > 1 and countvarsites >= maxSNPs:
                countvarsites+=1

        # pad with 4s as needed
        while matrix_out.shape[0] < maxSNPs:
            temparray=[4]*numTaxa
            matrix_out = np.append(matrix_out, np.array([temparray]), axis=0)

        # transpose genotype matrix so individuals: rows, SNPS: columns
        matrix_out = np.transpose(matrix_out)

        combos = list(itertools.combinations(range(matrix_out.shape[0]), 4))
        combo_rows = []
        for i in combos:
            combo_rows.append(matrix_out[i,:])
        matrix_full = np.concatenate(combo_rows)

        # calculate proportion invariant sites
        prop_inv = [1- (countvarsites / Length)]

        all_empirical_regions.append(matrix_full)
        all_empirical_pinv.append(prop_inv)

    all_empirical_regions = np.array(all_empirical_regions)
    all_empirical_pinv = np.array(all_empirical_pinv)

    return(all_empirical_regions, all_empirical_pinv)


def get_start_tree(start_tree, birthRate, IQ_Tree, Model, Input_Alignment, Input_Order, Input_Match_dict, Temp, Results):
    if start_tree == 'NJ':
        print('Inferring Neighbor Joining tree to use as start tree.\n')
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(Input_Alignment)
        constructor = DistanceTreeConstructor()
        njtree = constructor.nj(dm)
        io = StringIO()
        Phylo.write([njtree], io, "newick", plain=True)

        string = io.getvalue()
        string2 = re.sub('Inner\d+', '', string)
        string2 = add_bl(string2, birthRate)
        start_tree_file = open('%s/NJ.tre' % Results, 'w')
        start_tree_file.write(string2)
        start_tree_file.close()
        return(string2)

    elif start_tree == 'Random':
        print('Generating a random start tree.')
        random_simulator = simulators.Simulator(IQ_Tree, Model, 1, 1, Input_Alignment, Input_Order, Input_Match_dict, Temp)
        start_tree = random_simulator.simulateStartTree(birthRate)
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
