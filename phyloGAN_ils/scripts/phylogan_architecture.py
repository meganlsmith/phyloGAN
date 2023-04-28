#import functions
from training import *
import random
from discriminator import *
import datetime
import treemoves
import ete3
from parameter_proposals import *
import utils

class phyloGAN(object):

    def __init__(self, IQ_Tree, Model, Length, Chunks, Input_Order,
        Input_Match_dict, Results, Temp, true_tree, checkpoint, empirical_alignments, empirical_pinv, empirical_lengths):

        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Results = Results
        self.Length = Length
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp
        self.true_tree = true_tree
        self.empirical_alignments = empirical_alignments
        self.empirical_pinv = empirical_pinv
        self.empirical_lengths = empirical_lengths
        try:
            os.rmdir('%s' % self.Temp)
        except OSError:
            pass
        self.checkpoint = checkpoint


    def stage1(self, Min_scale, Max_scale, Min_coal, Max_coal, Width_scale, Width_coal, Iterations, Proposals):

        # get starting values
        current_scale = random.uniform(Min_scale, Max_scale)
        current_coal = random.uniform(Min_coal, Max_coal)
       # initiate trainer and a proposer
        Stage1_Trainer = Simple_Training(IQ_Tree = self.IQ_Tree, Model = self.Model,
            Chunks = self.Chunks, Length = self.Length,
            Input_Alignments = self.empirical_alignments, Input_Lengths = self.empirical_lengths, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp)
        My_proposer = Proposer()

        # initiate current distance and lists for storing results
        dist_curr = float('inf')
        current_scale_list = []
        current_coal_list = []
        current_distances_list = []


        # do training
        for i in range(Iterations):
            print("ITER", i)
            print("time", datetime.datetime.now().time())

            # calculate temperature
            T = utils.temperature(i, Iterations)

            # propose changes
            proposed_scale_best = None
            proposed_coal_best = None
            proposed_dist_best = float('inf')


            for rep in range(0, Proposals):

                # decide whether to propose new lambda or coal
                selected_proposal = np.random.choice([1,2])
                if selected_proposal == 1:
                    proposed_scale = My_proposer.proposal(curr_value = current_scale,
                        multiplier = T, proposal_width = Width_scale,
                        proposal_min = Min_scale, proposal_max = Max_scale)
                    proposed_coal = current_coal
                else:
                    proposed_coal = My_proposer.proposal(curr_value = current_coal,
                        multiplier = T, proposal_width = Width_coal,
                        proposal_min = Min_coal, proposal_max = Max_coal)
                    proposed_scale = current_scale
                # calculate the distance between the average proportions of invariant sites
                proposed_dist = Stage1_Trainer.simple_loss_stage1(proposed_scale, proposed_coal)

                if proposed_dist < proposed_dist_best:
                    print("keeping proposal", proposed_scale, proposed_coal, proposed_dist)
                    proposed_dist_best = proposed_dist
                    proposed_scale_best = proposed_scale
                    proposed_coal_best = proposed_coal

            if proposed_dist_best <= dist_curr:
                p_accept = 1
            else:
                p_accept =  dist_curr/proposed_dist_best*T

            rand = np.random.rand()
            accept = rand < p_accept

            if accept:
                print("Accepted scale ", proposed_scale_best, ', accepted coal ', proposed_coal_best, 'because ', p_accept)
                dist_curr = proposed_dist_best
                current_scale = proposed_scale_best
                current_coal = proposed_coal_best

            current_scale_list.append(current_scale)
            current_coal_list.append(current_coal)
            current_distances_list.append(dist_curr)
            print('Current distance: ', dist_curr, " Current scale: ", current_scale, "Current coal: ", current_coal, '\n')

        utils.write_results_stage1(self.Results, current_scale_list, current_coal_list, current_distances_list)
        return (current_scale, current_coal )


    def stage2(self, Pretraining_Iterations, Pretraining_Epochs, Training_Iterations, Training_Epochs, MaxTrees, inferred_scale, inferred_coal, start_tree, start_iter, checkpoint_interval):

        the_discriminator = discriminator()
        
        if self.checkpoint == True:
            the_discriminator.load_weights('./%s/Checkpoint_model' % self.Results)
            discriminator_state = the_discriminator
            
            # get Max SNPs
            MaxSNPs = int(open(self.Results + '/Checkpoint_MaxSNPs.txt', 'r').readlines()[-1].strip())

        else:
            # calculate max SNPs
            MaxSNPs = utils.calc_max_snps(self.empirical_alignments, self.Length, self.Chunks)


        # get Stage2_Trainer
        Stage2_Trainer = Training(discriminator = the_discriminator, IQ_Tree = self.IQ_Tree,
            Model = self.Model, Chunks = self.Chunks, Length = self.Length,
            Input_Alignments = self.empirical_alignments, Input_Lengths = self.empirical_lengths, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp,
            coal = inferred_coal, scale=inferred_scale, numPretrainingEpochs = Pretraining_Epochs,
            numTrainingEpochs = Training_Epochs, maxSNPs = MaxSNPs)


        if self.checkpoint == True:
            print('Continuing from checkpoint. Skip pretraining.\n')
            current_tree = start_tree
            current_loss = float(open(self.Results + '/Checkpoint_GeneratorLoss.txt', 'r').readlines()[-1].strip())
            current_accuracy = float(open(self.Results + '/Checkpoint_GeneratorFakeAcc.txt', 'r').readlines()[-1].strip())
            disc_loss = float(open(self.Results + '/Checkpoint_DiscriminatorLoss.txt', 'r').readlines()[-1].strip())
            real_acc = float(open(self.Results + '/Checkpoint_DiscriminatorRealAcc.txt', 'r').readlines()[-1].strip())
            fake_acc = float(open(self.Results + '/Checkpoint_DiscriminatorFakeAcc.txt', 'r').readlines()[-1].strip())

            # lists to store results
            results_trees = open(self.Results + '/Checkpoint_trees.txt', 'r').readlines()
            generator_loss = open(self.Results + '/Checkpoint_GeneratorLoss.txt', 'r').readlines()
            generator_fake_acc = open(self.Results + '/Checkpoint_GeneratorFakeAcc.txt', 'r').readlines()
            discriminator_loss = open(self.Results + '/Checkpoint_DiscriminatorLoss.txt', 'r').readlines()
            discriminator_real_acc = open(self.Results + '/Checkpoint_DiscriminatorRealAcc.txt', 'r').readlines()
            discriminator_fake_acc = open(self.Results + '/Checkpoint_DiscriminatorFakeAcc.txt', 'r').readlines()

            results_trees = [x.strip() for x in results_trees]
            generator_loss = [x.strip() for x in generator_loss]
            generator_fake_acc = [x.strip() for x in generator_fake_acc]
            discriminator_loss = [x.strip() for x in discriminator_loss]
            discriminator_real_acc = [x.strip() for x in discriminator_real_acc]
            discriminator_fake_acc = [x.strip() for x in discriminator_fake_acc]

        
        else:
            
            

            # pretraining
            print('Begin pre-training.\n')
    
            
    
            for i in range(Pretraining_Iterations):
    
                # store max accuracy and min losses
                max_acc = 0
                max_loss = 0
                min_loss = float('inf')
    
                # simulate data under some tree topology with BL based on inferred lambda
                real_acc, fake_acc, disc_loss, tree = Stage2_Trainer.pretraining_stage2()
    
                avg_acc = (real_acc + fake_acc) / 2
                if avg_acc > max_acc:
                    max_acc = avg_acc
                    min_loss = disc_loss
                if disc_loss > max_loss:
                    max_loss = disc_loss
                    best_tree = tree
    
            print('\n\nMaximum accuracy achieved: ', max_acc, ' Best loss: ', min_loss)

            print('Begin training.')
    
            current_loss = float('inf')
            current_accuracy = 0
            current_tree = start_tree
    
            # lists to store results
            results_trees = []
            generator_loss = []
            generator_fake_acc = []
            discriminator_loss = []
            discriminator_real_acc = []
            discriminator_fake_acc = []
    
        for i in range(start_iter, Training_Iterations):
            print('\nCurrent loss: ', current_loss, " Current tree: ", current_tree)
            print("ITER", i)
            print("time", datetime.datetime.now().time())

            # calculate temperature
            T = utils.temperature(i, Training_Iterations)

            # if i == 0, get start trees
            if i == 0:
                trees = [start_tree]

            else:
                mover = treemoves.TreeMove(current_tree, T, MaxTrees, inferred_coal)
                trees = mover()

            # set up variables
            proposed_loss_best = float('inf')
            proposed_acc_best = 0
            proposed_tree_best = None

            for proposed_tree in trees:

                # calculate the generator loss
                proposed_loss, proposed_accuracy = Stage2_Trainer.generator_loss_stage2(proposed_tree)
                if proposed_loss < proposed_loss_best:
                    proposed_loss_best = proposed_loss
                    proposed_acc_best = proposed_accuracy
                    proposed_tree_best = proposed_tree
                    print('Keep tree', proposed_tree, ' because loss ', proposed_loss, ' and accuracy ', proposed_accuracy)

            if proposed_loss_best < current_loss:
                p_accept = 1

            else:
                p_accept =  current_loss/proposed_loss_best*T*0.25
                print(p_accept, T)
            rand = np.random.rand()
            accept = rand < p_accept


            # if accept, retrain
            if accept:
                print("Accepted tree ", proposed_tree_best, ' because ', p_accept)
                current_loss = proposed_loss_best
                current_accuracy = proposed_acc_best
                current_tree = proposed_tree_best


                # train
                real_acc, fake_acc, disc_loss, discriminator_state = Stage2_Trainer.train_sa_stage2(proposed_tree)
            

            # add info to lists
            results_trees.append(current_tree)
            generator_loss.append(current_loss)
            generator_fake_acc.append(current_accuracy)
            discriminator_loss.append(float(disc_loss))
            discriminator_real_acc.append(real_acc)
            discriminator_fake_acc.append(fake_acc)

            # recalculate loss 
            #recalc_loss, recalc_accuracy = Stage2_Trainer.generator_loss_stage2(current_tree)
            #current_loss = recalc_loss
            
            # log if iteration is multiple of 100 (do 10 for testing, then change to 100)

            if (i+1) % int(checkpoint_interval) == 0:
                print('creating checkpoint files...')
                utils.create_checkpoint(results_trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, self.Results, MaxSNPs)
                discriminator_state.save_weights(filepath = './%s/Checkpoint_model' % self.Results)

        # write results
        utils.write_results_stage2(results_trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, self.Results, self.true_tree)



