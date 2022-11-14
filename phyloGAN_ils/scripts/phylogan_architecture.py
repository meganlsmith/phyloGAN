import parameter_proposals
import utils
import random
import numpy as np
import simulators
import discriminator
import training
import dendropy
import treemoves

class stage1(object):

    def __init__(self, Min_Coal, Max_Coal, Width_Coal, Min_Scale, Max_Scale, Width_Scale, 
            Iterations, Proposals, Empirical_PIS, Empirical_Lengths, Empirical_Character_Matrices,
            Temp_Dir, Num_Taxa, IQ_Tree, Gene_Tree_Samples, Model, Results, match_dict):
            
        self.Min_Coal = Min_Coal
        self.Max_Coal = Max_Coal
        self.Width_Coal = Width_Coal
        self.Min_Scale = Min_Scale
        self.Max_Scale = Max_Scale
        self.Width_Scale = Width_Scale
        self.Iterations = Iterations
        self.Proposals = Proposals
        self.Empirical_PIS = Empirical_PIS
        self.Empirical_Lengths = Empirical_Lengths
        self.Empirical_Character_Matrices = Empirical_Character_Matrices
        self.Temp_Dir = Temp_Dir
        self.Num_Taxa = Num_Taxa
        self.IQ_Tree = IQ_Tree
        self.Gene_Tree_Samples = Gene_Tree_Samples
        self.Model = Model
        self.Results = Results
        self.match_dict = match_dict

    def sample_empirical(self):
    
        sampled_alignments = np.random.choice(a=range(len(self.Empirical_PIS)), size = self.Gene_Tree_Samples, replace = True)
        empirical_lengths = np.array(self.Empirical_Lengths)
        empirical_pis = np.array(self.Empirical_PIS)
        sampled_lengths = empirical_lengths[sampled_alignments]
        sampled_pis = empirical_pis[sampled_alignments]
        mean_sampled_pis = sum(sampled_pis)/len(sampled_pis)
        std_sampled_pis = np.std(sampled_pis)
        sampled_charmatrices = [self.Empirical_Character_Matrices[x] for x in sampled_alignments]
    
        return(mean_sampled_pis, std_sampled_pis, sampled_lengths, sampled_charmatrices)

    def __call__(self):
    
        # starting values for coal_time, scale
        coal_current = random.uniform(self.Min_Coal, self.Max_Coal)
        scale_current = random.uniform(self.Min_Scale, self.Max_Scale)
        distance_current = float('inf')
        My_proposer = parameter_proposals.Proposer()
        Simulator = simulators.Simulator(self.Temp_Dir, self.Num_Taxa, self.IQ_Tree, self.Model, self.match_dict)
        
        coal_list = []
        scale_list = []
        
        for i in range(self.Iterations):
        
        
            print('\nIteration ', i)
            print('Current distance: ', distance_current)
        
            # calculate temperature
            T = utils.temperature(i, self.Iterations)
        
            best_proposed_distance = float('inf')
        
            for j in range(self.Proposals):
        
                # randomly select a change in coal time or scale
                selected_proposal = np.random.choice([1,2])
                
                if selected_proposal == 1:
        
                    # propose a new coal_time
                    proposed_coal = My_proposer.proposal(curr_value = coal_current,
                        multiplier = T, proposal_width = self.Width_Coal,
                        proposal_min = self.Min_Coal, proposal_max = self.Max_Coal)
                    proposed_scale = scale_current
        
                else:
        
                    # propose a new coal_time
                    proposed_scale = My_proposer.proposal(curr_value = scale_current,
                        multiplier = T, proposal_width = self.Width_Scale,
                        proposal_min = self.Min_Scale, proposal_max = self.Max_Scale)
                    proposed_coal = coal_current
            
        
                # sample empirical data
                mean_sampled_pis, std_sampled_pis, sampled_empirical_lengths, sampled_charmatrices = self.sample_empirical()
                
                # simulate data to match empirical data in terms of lengths
                
                mean_simulated_pis, std_simulated_pis = Simulator.simulate_matching(sampled_empirical_lengths, proposed_coal, proposed_scale)

                # calculate distance
                distance = np.linalg.norm(np.array(mean_sampled_pis, std_sampled_pis) - np.array(mean_simulated_pis, std_simulated_pis))
                
                # decide whether this is the best current prosal
                if distance <= best_proposed_distance:
                    best_proposed_distance = distance
                    best_proposed_scale = proposed_scale
                    best_proposed_coal = proposed_coal
                    #print('Keep because %s, %s and %s, %s lead to distance %s. Coal: %s, Scale: %s' % (mean_sampled_pis, mean_simulated_pis, std_sampled_pis, std_simulated_pis, distance, proposed_coal, proposed_scale))
                
            # calculate p_accept
            if best_proposed_distance <= distance_current:
                p_accept = 1
            else:
                p_accept = distance_current/best_proposed_distance*T
            
            rand = np.random.rand()
            accept = rand < p_accept
            
            if accept:
                #print('accepted proposal with coal time %s and scale %s because of distance %s and prob %s' % (best_proposed_coal, best_proposed_scale, best_proposed_distance, p_accept))
                coal_current = best_proposed_coal
                scale_current = best_proposed_scale
                distance_current = best_proposed_distance

            coal_list.append(coal_current)
            scale_list.append(scale_current)
        
        print("\nRESULTS:\nCoalescent time: %s\nScale: %s" % (coal_current, scale_current))
        
        utils.write_results_stage1(coal_list, scale_list, self.Results)

        return(coal_current, scale_current)
        

class stage2(object):

    def __init__(self, empirical_var, IQTree_path, Batch_Size, Pretraining_Iterations, Gene_tree_samples,
        empirical_pis, empirical_lengths, empirical_charmatrices, taxon_order,
        Temp_Dir, Num_Taxa, inferred_coal, inferred_scale, Pretraining_Epochs, Training_Epochs,
        Training_Iterations, Max_trees, Results, True_tree, Model, start_tree, check_point, start_from_ckpt, match_dict):
            
        self.empirical_var = empirical_var
        self.IQTree_path = IQTree_path
        self.Batch_Size = Batch_Size
        self.Pretraining_Iterations = Pretraining_Iterations
        self.Gene_tree_samples = Gene_tree_samples
        self.empirical_pis = empirical_pis
        self.empirical_lengths = empirical_lengths
        self.empirical_charmatrices =empirical_charmatrices
        self.taxon_order = taxon_order
        self.Temp_Dir = Temp_Dir
        self.Num_Taxa = Num_Taxa
        self.inferred_coal = inferred_coal
        self.inferred_scale = inferred_scale
        self.Pretraining_Epochs = Pretraining_Epochs
        self.Training_Epochs = Training_Epochs
        self.Training_Iterations = Training_Iterations
        self.Max_trees = Max_trees
        self.Results = Results
        self.True_tree = True_tree
        self.Model = Model
        self.start_tree = start_tree
        self.discriminator = discriminator.discriminator()
        self.Simulator = simulators.Simulator(Temp_Dir, Num_Taxa, IQTree_path, Model, match_dict)
        self.check_point = check_point
        self.start_from_ckpt = start_from_ckpt
        self.match_dict = match_dict

    def sample_empirical_batches(self):
    
        all_mean_sampled_pis = []
        all_std_sampled_pis = []
        all_sampled_lengths = []
        all_sampled_charmatrices = []
    
        for i in range(self.Batch_Size):
            sampled_alignments = np.random.choice(a=range(len(self.empirical_pis)), size = self.Gene_tree_samples, replace = True)
            empirical_lengths = np.array(self.empirical_lengths)
            empirical_pis = np.array(self.empirical_pis)
            sampled_lengths = empirical_lengths[sampled_alignments]
            sampled_pis = empirical_pis[sampled_alignments]
            mean_sampled_pis = sum(sampled_pis)/len(sampled_pis)
            std_sampled_pis = np.std(sampled_pis)
            sampled_charmatrices = [self.empirical_charmatrices[x] for x in sampled_alignments]
    
            all_mean_sampled_pis.append(mean_sampled_pis)
            all_std_sampled_pis.append(std_sampled_pis)
            all_sampled_lengths.append(sampled_lengths)
            all_sampled_charmatrices.append(sampled_charmatrices)


        return(all_mean_sampled_pis, all_std_sampled_pis, all_sampled_lengths, all_sampled_charmatrices)

    def empirical_tonparray(self, char_matrix_list, max_var):
        # list to hold all results across reps
        all_generated_regions = []
        all_generated_pvar = []
        for i in range(self.Batch_Size):
            thematrix, thevar = utils.charmatrix_to_nparray(char_matrix_list[i], max_var, self.Num_Taxa, self.taxon_order)
            all_generated_regions.append(thematrix)
            all_generated_pvar.append(thevar)
    
        all_generated_regions = np.array(all_generated_regions)
        all_generated_pvar = np.array(all_generated_pvar)
    
        return(all_generated_regions, all_generated_pvar)

    def __call__(self):
        
        if self.start_from_ckpt == True:
            print('Starting from checkpoint.')
            start_tree_str = open('%s/Checkpoint_trees.txt' % self.Results, 'r').readlines()[-1].strip()
            the_start_tree = dendropy.Tree.get(data=start_tree_str, schema="newick")
            mean_empirical_var = float(open('%s/Checkpoint_MaxSNPs.txt' % self.Results, 'r').readlines()[-1].strip())
            beginning_iter = len(open('%s/Checkpoint_trees.txt' % self.Results, 'r').readlines())
            current_tree = the_start_tree
            
            current_loss = float(open('%s/Checkpoint_GeneratorLoss.txt' % self.Results, 'r').readlines()[-1].strip())
            current_fakeacc = float(open('%s/Checkpoint_GeneratorFakeAcc.txt' % self.Results, 'r').readlines()[-1].strip())
            disc_loss = float(open('%s/Checkpoint_DiscriminatorLoss.txt' % self.Results, 'r').readlines()[-1].strip())
            real_acc = float(open('%s/Checkpoint_DiscriminatorRealAcc.txt' % self.Results, 'r').readlines()[-1].strip())
            fake_acc = float(open('%s/Checkpoint_DiscriminatorFakeAcc.txt' % self.Results, 'r').readlines()[-1].strip())

            # lists to store results
            current_tree_list = open(self.Results + '/Checkpoint_trees.txt', 'r').readlines()
            generator_loss_list = open(self.Results + '/Checkpoint_GeneratorLoss.txt', 'r').readlines()
            generator_fakeacc_list = open(self.Results + '/Checkpoint_GeneratorFakeAcc.txt', 'r').readlines()
            discriminator_loss_list = open(self.Results + '/Checkpoint_DiscriminatorLoss.txt', 'r').readlines()
            discriminator_realacc_list = open(self.Results + '/Checkpoint_DiscriminatorRealAcc.txt', 'r').readlines()
            discriminator_fakeacc_list = open(self.Results + '/Checkpoint_DiscriminatorFakeAcc.txt', 'r').readlines()

            current_tree_list = [x.strip() for x in current_tree_list]
            generator_loss_list = [x.strip() for x in generator_loss_list]
            generator_fakeacc_list = [x.strip() for x in generator_fakeacc_list]
            discriminator_loss_list = [x.strip() for x in discriminator_loss_list]
            discriminator_realacc_list = [x.strip() for x in discriminator_realacc_list]
            discriminator_fakeacc_list = [x.strip() for x in discriminator_fakeacc_list]

            # load disc weights
            self.discriminator.load_weights('./%s/Checkpoint_model' % self.Results)
            discriminator_state = self.discriminator

            # get trainer
            MyTrainer = training.Training(self.discriminator, self.IQTree_path, mean_empirical_var, self.Batch_Size)

        else:
            # start tree
            print('Start with a %s tree.' % self.start_tree)
            
            if self.start_tree == 'Random':
                the_start_tree = dendropy.simulate.treesim.birth_death_tree(birth_rate = 1, death_rate = 0, num_extant_tips = self.Num_Taxa).as_string(schema="newick", suppress_rooting=True)
                the_start_tree = utils.replace_names(tree_string = the_start_tree, Input_Match_dict = self.match_dict)
                print(the_start_tree)
            else:
                the_start_tree = dendropy.Tree.get(path=self.start_tree, schema="newick")
            
            # number of variable sites to keep
            mean_empirical_var = sum(self.empirical_var)/len(self.empirical_var)
            print('Keep %r SNPs.' % mean_empirical_var)
        
            # pretraining
            MyTrainer = training.Training(self.discriminator, self.IQTree_path, mean_empirical_var, self.Batch_Size)
            
            print("\nBeginning pretraining iterations!\n")
        
            for iteration in range(self.Pretraining_Iterations):
                # sample the empirical data n batches times
                mean_sampled_pis, std_sampled_pis, sampled_empirical_lengths, sampled_charmatrices = self.sample_empirical_batches()
                
                # convert character matrices to numpy array
                empirical_matrix_full, empirical_variable_sites = self.empirical_tonparray(sampled_charmatrices, mean_empirical_var)
                
                # simulate data under a random tree with coalescent times and scaling factors inferred in Step 1.
                simulated_matrix_full, simulated_variable_sites = self.Simulator.simulate_matching_stage2(sampled_empirical_lengths, mean_empirical_var, self.inferred_coal, self.inferred_scale, self.Batch_Size, self.taxon_order)
    
                # train the discriminator
                real_acc, fake_acc, disc_loss, discriminator_state = MyTrainer.train_sa_stage2(self.Pretraining_Epochs, simulated_matrix_full, simulated_variable_sites, empirical_matrix_full, empirical_variable_sites)
                
                print('\nIteration %s complete. ' % iteration)
                print('Real_acc: %r ' % real_acc)
                print('Fake_acc: %r\n ' % fake_acc)
    
    
                generator_loss_list = []
                discriminator_loss_list = []
                current_tree_list = []
                generator_fakeacc_list = []
                discriminator_realacc_list = []
                discriminator_fakeacc_list = []
                current_loss = float('inf')
                
                beginning_iter = 0

    
        # training
        print("Beginning training iterations!\n")
    
        for iteration in range(beginning_iter, self.Training_Iterations):
        
            # calculate temperature
            T = utils.temperature(iteration, self.Training_Iterations)
    
            # set up for proposals
            proposed_loss_best = float('inf')
            proposed_fakeacc_best = 0.0
    
            if iteration == 0:
                
                #random start tree
                current_tree = the_start_tree
                proposed_tree_list = [current_tree]
    
            else:
                
                # propose trees
                if type(current_tree)!=str:
                    current_tree = current_tree.as_string(schema="newick",suppress_rooting=True)
                mover = treemoves.TreeMove(current_tree, T, self.Max_trees, self.inferred_coal, self.match_dict)
                proposed_tree_list = mover()
        
            proposed_loss_best = float('inf')
            proposed_fakeacc_best = 0
    
            for proposed_tree in proposed_tree_list:
            
                # simulate data under the tree after getting lengths from empirical data
                mean_sampled_pis, std_sampled_pis, proposed_sampled_empirical_lengths, proposed_sampled_empirical_charmatrices = self.sample_empirical_batches()
                
                # simulate data
                proposed_generated_regions, proposed_generated_var = self.Simulator.simulate_matching_stage2_training(proposed_sampled_empirical_lengths, self.inferred_coal,
                    self.inferred_scale, self.taxon_order, mean_empirical_var, self.Batch_Size, proposed_tree)
                
                proposed_loss, proposed_fakeacc = MyTrainer.generator_loss(proposed_generated_regions, proposed_generated_var)
    
                #do we keep this proposal
                if proposed_loss <= proposed_loss_best:
                    proposed_loss_best = proposed_loss
                    proposed_fakeacc_best = proposed_fakeacc
                    proposed_tree_best = proposed_tree
                    proposed_generated_regions_best = proposed_generated_regions
                    proposed_generated_var_best = proposed_generated_var
                    proposed_sampled_empirical_lengths_best = proposed_sampled_empirical_lengths
                    proposed_sampled_empirical_charmatrices_best = proposed_sampled_empirical_charmatrices
                    #print('Keep tree %s because %r and %r' % (proposed_tree, proposed_loss, proposed_fakeacc))
    
        
        
            if proposed_loss_best <= current_loss:
                p_accept = 1
            
            else:
                p_accept = current_loss/proposed_loss_best*T
         
            rand = np.random.rand()
        
            accept = rand < p_accept
            
            if accept:
                #print('Accepted tree %s because of loss %s and fake accuracy %s and accept prob %s' % (proposed_tree_best, proposed_loss_best, proposed_fakeacc_best, p_accept))
                regions_current = proposed_generated_regions_best
                var_current = proposed_generated_var_best
                current_tree = proposed_tree_best
                current_loss = proposed_loss_best
                current_fakeacc = proposed_fakeacc_best
                
        
                ## train the discriminator
                ## convert character matrices to numpy array
                empirical_matrix_full, empirical_variable_sites = self.empirical_tonparray(proposed_sampled_empirical_charmatrices_best, mean_empirical_var)
                real_acc, fake_acc, disc_loss, discriminator_state = MyTrainer.train_sa_stage2(self.Training_Epochs, regions_current, var_current, empirical_matrix_full, empirical_variable_sites)
            
            # add results to list
            generator_loss_list.append(current_loss)
            generator_fakeacc_list.append(current_fakeacc)
            discriminator_loss_list.append(disc_loss)
            discriminator_realacc_list.append(real_acc)
            discriminator_fakeacc_list.append(fake_acc)
            
            # recalculate loss
             # simulate data
             # simulate data under the tree after getting lengths from empirical data
            recalc_mean_sampled_pis, recalc_std_sampled_pis, recalc_sampled_empirical_lengths, recalc_sampled_empirical_charmatrices = self.sample_empirical_batches()
            recalc_generated_regions, recalc_generated_var = self.Simulator.simulate_matching_stage2_training(recalc_sampled_empirical_lengths, self.inferred_coal,
                    self.inferred_scale, self.taxon_order, mean_empirical_var, self.Batch_Size, current_tree)
            recalc_loss, recalc_accuracy = MyTrainer.generator_loss(recalc_generated_regions, recalc_generated_var)
            
            
            if type(current_tree) == str:
                current_tree_list.append(current_tree)
            else:
                current_tree_list.append(current_tree.as_string(schema="newick", suppress_rooting=True))
            
            
            if (iteration+1) % int(self.check_point) == 0:
                #print('creating checkpoint files...')
                utils.create_checkpoint(current_tree_list, generator_loss_list, generator_fakeacc_list, discriminator_loss_list, discriminator_realacc_list, discriminator_fakeacc_list, self.Results, mean_empirical_var)
                discriminator_state.save_weights('./%s/Checkpoint_model' % self.Results)

            print('\nIteration %s complete. ' % iteration)
            print('Real_acc: %r ' % real_acc)
            print('Fake_acc: %r ' % fake_acc)
            print('Tree: %s\n ' % current_tree)
        
        utils.write_results_stage2(current_tree_list, generator_loss_list, generator_fakeacc_list, discriminator_loss_list, discriminator_realacc_list, discriminator_fakeacc_list, self.Results, self.True_tree)
    
    
    
