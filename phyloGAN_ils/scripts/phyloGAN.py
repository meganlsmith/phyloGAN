from phylogan_architecture import *
import sys
from datetime import datetime
import utils
import os

def main():

    # read params file
    args = sys.argv[1:]
    parameters = utils.read_params_file(args[0])

    os.system('mkdir %s' % parameters['results'])
    
    # read empirical data
    empirical_parser = utils.Parse_Empirical(parameters['alignment_folder'])
    empirical_charmatrices, empirical_pinv, empirical_lengths, num_taxa, taxon_order = empirical_parser()
    # map alignment to alisim names
    match_dict = utils.create_matchdict(taxon_order)

    # decide whether to run full, generator only, or discriminator only (for debugging)
    if len(args) == 1: # then run the full inference.

        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Order = taxon_order,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = False, empirical_alignments = empirical_charmatrices,
            empirical_pinv = empirical_pinv, empirical_lengths=empirical_lengths) 
    
        # stage 1: infer lambda
        print('\n\nSTAGE 1: Infer LAMBDA.\n\n')
        estimated_scale, estimated_coal = my_phylogan.stage1(Min_scale = parameters['Min_scale'], Max_scale = parameters['Max_scale'],
            Min_coal = parameters['Min_coal'], Max_coal = parameters['Max_coal'], Width_scale = parameters['Width_scale'],
            Width_coal = parameters['Width_coal'], Iterations = parameters['Stage_1_Iterations'],
	    Proposals = parameters['Stage_1_Proposals'])
    

        # get start tree
        start_tree = utils.get_start_tree(start_tree = parameters['start_tree'],
            coal = estimated_coal,
            Input_Alignments = empirical_charmatrices, Results = parameters['results'],
            Input_Order = taxon_order, Input_Match_dict = match_dict)

        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated coal = %r; Estimated scale = %r\n\n' % (estimated_coal, estimated_scale))
        my_phylogan.stage2(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_scale = estimated_scale, inferred_coal = estimated_coal,
            start_tree = start_tree, start_iter = 0, checkpoint_interval = parameters['checkpoint_interval'])

        print('Phylogan run complete!')

    elif args[1] == 'checkpoint': # then continue from a checkpoint.

        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Order = taxon_order,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = True, empirical_alignments = empirical_charmatrices,
            empirical_pinv = empirical_pinv, empirical_lengths=empirical_lengths)
    
        # stage 1: get estimated lambda
        print('\n\nSTAGE 1: Get COAL and SCALE from previous run.\n\n')
        try:
            estimated_coal = float(open('%s/Coal.txt' % parameters['results'], 'r').readlines()[-1].strip())
            estimated_scale = float(open('%s/Scales.txt' % parameters['results'], 'r').readlines()[-1].strip())
        except:
            sys.exit('ERROR: You specified to run from a checkpoint, but it does not appear that Stage 1 completed. Please run without the "checkpoint" flag.')

        # get start tree
        start_tree = open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()[-1].strip()
        print ('Start from iteraton %r.\n' % len(open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()))
        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated coal = %r; Estimated scale = %r\n\n' % (estimated_coal, estimated_scale))

        my_phylogan.stage2(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_scale = estimated_scale, inferred_coal = estimated_coal,
            start_tree = start_tree, start_iter = len(open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()), checkpoint_interval = parameters['checkpoint_interval'])


        print('Phylogan run complete!')


if __name__ == "__main__":
    start = datetime.now()
    main()
    difference = datetime.now() - start
    print(difference)
