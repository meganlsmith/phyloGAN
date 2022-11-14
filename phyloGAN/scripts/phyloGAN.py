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
    
    try:
        pseudodir = parameters['Output_pseudoobserved'].split('/')[:-1]
        os.system('mkdir %s' % '/'.join(pseudodir))
    except:
        pass
    
    # if Simulate_pseudoobserved is true, simulate the data
    if len(parameters.keys())== 25 and (len(args) == 1 or args[1] != 'checkpoint'):
        utils.simulatePseudo(iqTree = parameters['IQTree_path'], birthRate = parameters['Birth_rate'],
            model = parameters['Simulation_nucleotide_substitution_model'], numTaxa = parameters['Num_taxa'],
            length = parameters['Full_length'], output = parameters['Output_pseudoobserved'])
        print(parameters['Output_pseudoobserved'])
        # read input alignment
        align, sporder = utils.readPhylip(parameters['Output_pseudoobserved'] + '.phy')

        # map alignment to alisim names
        match_dict = utils.create_matchdict(sporder)

    elif len(parameters.keys())== 25 and args[1] == 'checkpoint':
        # read input alignment
        align, sporder = utils.readPhylip(parameters['Output_pseudoobserved'] + '.phy')

        # map alignment to alisim names
        match_dict = utils.create_matchdict(sporder)


    else:
        # read input alignment
        align, sporder = utils.readPhylip(parameters['alignment_file'])

        # map alignment to alisim names
        match_dict = utils.create_matchdict(sporder)


    # decide whether to run full, generator only, or discriminator only (for debugging)
    if len(args) == 1: # then run the full inference.

        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Alignment = align, Input_Order = sporder,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = False)
    
        # stage 1: infer lambda
        print('\n\nSTAGE 1: Infer LAMBDA.\n\n')
        estimated_lambda = my_phylogan.stage1(Min_lambda = parameters['Min_lambda'], Max_lambda = parameters['Max_lambda'],
            Width_lambda = parameters['Width_lambda'], Iterations = parameters['Stage_1_Iterations'],
            Proposals = parameters['Stage_1_Proposals'])
    
        # get start tree
        start_tree = utils.get_start_tree(start_tree = parameters['start_tree'],
            birthRate = estimated_lambda, IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Input_Alignment = align,
            Input_Order = sporder, Input_Match_dict = match_dict, Temp = parameters['temp'], Results = parameters['results'])

        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated lambda = %r\n\n' % estimated_lambda)
        my_phylogan.stage2(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_lambda = estimated_lambda, start_tree = start_tree, start_iter = 0, checkpoint_interval = parameters['checkpoint_interval'])

        print('Phylogan run complete!')

    elif args[1] == 'checkpoint': # then continue from a checkpoint.

        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Alignment = align, Input_Order = sporder,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = True)
    
        # stage 1: get estimated lambda
        print('\n\nSTAGE 1: Get LAMBDA from previous run.\n\n')
        try:
            estimated_lambda = float(open('%s/Lambdas.txt' % parameters['results'], 'r').readlines()[-1].strip())
        except:
            sys.exit('ERROR: You specified to run from a checkpoint, but it does not appear that Stage 1 completed. Please run without the "checkpoint" flag.')

        # get start tree
        start_tree = open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()[-1].strip()
        print ('Start from iteraton %r.\n' % len(open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()))
        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated lambda = %r\n\n' % estimated_lambda)
        my_phylogan.stage2(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_lambda = estimated_lambda, start_tree = start_tree, start_iter = len(open('%s/Checkpoint_trees.txt' % parameters['results'], 'r').readlines()), checkpoint_interval = parameters['checkpoint_interval'])

        print('Phylogan run complete!')

    elif args[1] == 'generator': # run generator-only version
        print('Testing the generator.')
        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Alignment = align, Input_Order = sporder,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = False)
    
        # stage 1: infer lambda
        print('\n\nSTAGE 1: Infer LAMBDA.\n\n')
        estimated_lambda = my_phylogan.stage1(Min_lambda = parameters['Min_lambda'], Max_lambda = parameters['Max_lambda'],
            Width_lambda = parameters['Width_lambda'], Iterations = parameters['Stage_1_Iterations'],
            Proposals = parameters['Stage_1_Proposals'])
    
        # get start tree
        start_tree = utils.get_start_tree(start_tree = parameters['start_tree'],
            birthRate = estimated_lambda, IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Input_Alignment = align,
            Input_Order = sporder, Input_Match_dict = match_dict, Temp = parameters['temp'], Results = parameters['results'])
        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated lambda = %r\n\n' % estimated_lambda)
        my_phylogan.stage2_generator(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_lambda = estimated_lambda, start_iter = 0)

        print('Phylogan run complete!')

    elif args[1] == 'discriminator': # run the discriminator only version
        print('Testing the discriminator.')
        # initiate object of class phylogan
        my_phylogan = phyloGAN(IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Length = parameters['Length'],
            Chunks = parameters['Chunks'], Input_Alignment = align, Input_Order = sporder,
            Input_Match_dict = match_dict, Results = parameters['results'], Temp = parameters['temp'],
            true_tree = parameters['true_tree'], checkpoint = False)
    
        # stage 1: infer lambda
        print('\n\nSTAGE 1: Infer LAMBDA.\n\n')
        estimated_lambda = my_phylogan.stage1(Min_lambda = parameters['Min_lambda'], Max_lambda = parameters['Max_lambda'],
            Width_lambda = parameters['Width_lambda'], Iterations = parameters['Stage_1_Iterations'],
            Proposals = parameters['Stage_1_Proposals'])
    
        # get start tree
        start_tree = utils.get_start_tree(start_tree = parameters['start_tree'],
            birthRate = estimated_lambda, IQ_Tree = parameters['IQTree_path'],
            Model = parameters['Nucleotide_substitution_model'], Input_Alignment = align,
            Input_Order = sporder, Input_Match_dict = match_dict, Temp = parameters['temp'], Results = parameters['results'])
        # stage 2: infer topology
        print('\n\nSTAGE 2: Infer TOPOLOGY; Estimated lambda = %r\n\n' % estimated_lambda)
        my_phylogan.stage2_discriminator(Pretraining_Iterations = parameters['Stage_2_Pretraining_Iterations'],
            Pretraining_Epochs = parameters['Stage_2_Pretraining_Epochs'],
            Training_Iterations =  parameters['Stage_2_Training_Iterations'],
            Training_Epochs = parameters['Stage_2_Training_Epochs'],
            MaxTrees = parameters['Max_trees'],
            inferred_lambda = estimated_lambda, start_tree = start_tree, start_iter = 0)

        print('Phylogan run complete!')

if __name__ == "__main__":
    start = datetime.now()
    main()
    difference = datetime.now() - start
    print(difference)
