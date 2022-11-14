
# imports
import os
import utils
import sys
import simulators
import phylogan_architecture


def main():
    args = sys.argv[1:]
    parameters = utils.read_params_file(args[0])

    if parameters['Simulate_pseudoobserved'] == True and (len(args) == 1 or args[1] != 'checkpoint'):

        # Step 1: Simulate pseudoobserved data
        print('\nSIMULATING PSEUDO-OBSERVED DATA.')
        
        os.system('mkdir %s' % parameters['Output_pseudoobserved'])
        os.system('mkdir %s/gene_trees' % parameters['Output_pseudoobserved'])
        os.system('mkdir %s/alignments' % parameters['Output_pseudoobserved'])
        
        pseudosim = simulators.Pseudo_Simulator(parameters['Num_taxa'], parameters['Coal_time'], parameters['Output_pseudoobserved'], 
            parameters['Gene_trees'], parameters['Length'], parameters['Scale'], parameters['temp'], parameters['IQTree_path'], parameters['Nucleotide_substitution_model'])
        pis, std_pis = pseudosim()
        
    
    else:
        print("\nANALYZING EMPIRICAL DATA.")
    
    # Step 2: Infer optimal coalescent time and scale values
    """We need functions that will take empirical input, store the alignments and alignment lengths, and calculate the number of PIS and the stdev in this number of replicates of size N."""
    
    # read empirical data
    empirical_parser = utils.Parse_Empirical(parameters['alignment_folder'])
    empirical_charmatrices, empirical_pis, empirical_lengths, empirical_var, num_taxa, taxon_order = empirical_parser()
    # map alignment to alisim names
    match_dict = utils.create_matchdict(taxon_order)
    #print(match_dict)
    
    if len(args) > 1 and args[1] == 'checkpoint':
        print('\nSTARTING FROM CHECKPOINT.')
        
        # get coal time and scale
        inferred_coal = float(open('%s/CoalTime.txt' % parameters['results'], 'r').readlines()[-1].strip())
        inferred_scale = float(open('%s/Scale.txt' % parameters['results'], 'r').readlines()[-1].strip())

        # Stage2: Infer the species tree topology
        
        stage2 = phylogan_architecture.stage2(empirical_var, parameters['IQTree_path'], parameters['Batch_size'], 
            parameters['Stage_2_Pretraining_Iterations'], parameters['Gene_tree_samples'], 
            empirical_pis, empirical_lengths, empirical_charmatrices, taxon_order,
            parameters['temp'], num_taxa, inferred_coal, inferred_scale, parameters['Stage_2_Pretraining_Epochs'], parameters['Stage_2_Training_Epochs'],
            parameters['Stage_2_Training_Iterations'], parameters['Max_trees'], parameters['results'], parameters['true_tree'], parameters['Nucleotide_substitution_model'], parameters['start_tree'], parameters['check_point'], True, match_dict)
        stage2()

        
    
    else:
        # do stage 1
        stage1 = phylogan_architecture.stage1(parameters['Min_coal'], parameters['Max_coal'], parameters['Width_coal'], parameters['Min_scale'], parameters['Max_scale'], parameters['Width_scale'], 
                    parameters['Stage_1_Iterations'], parameters['Stage_1_Proposals'], empirical_pis, empirical_lengths, empirical_charmatrices,
                    parameters['temp'], num_taxa, parameters['IQTree_path'], parameters['Gene_tree_samples'], parameters['Nucleotide_substitution_model'], parameters['results'], match_dict)
        inferred_coal, inferred_scale = stage1()
        
        # Stage2: Infer the species tree topology
        
        stage2 = phylogan_architecture.stage2(empirical_var, parameters['IQTree_path'], parameters['Batch_size'], 
            parameters['Stage_2_Pretraining_Iterations'], parameters['Gene_tree_samples'], 
            empirical_pis, empirical_lengths, empirical_charmatrices, taxon_order,
            parameters['temp'], num_taxa, inferred_coal, inferred_scale, parameters['Stage_2_Pretraining_Epochs'], parameters['Stage_2_Training_Epochs'],
            parameters['Stage_2_Training_Iterations'], parameters['Max_trees'], parameters['results'], parameters['true_tree'], parameters['Nucleotide_substitution_model'], parameters['start_tree'], parameters['check_point'], False, match_dict)
        stage2()

if __name__ == "__main__":
    main()

    
