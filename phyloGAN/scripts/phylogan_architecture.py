"""Parts of this code are adapted from Wang et al., 2021, Molecular Ecology Resources, https://doi.org/10.1111/1755-0998.13386."""
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

    def __init__(self, IQ_Tree, Model, Length, Chunks, Input_Alignment, Input_Order,
        Input_Match_dict, Results, Temp, true_tree, checkpoint):

        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Input_Alignment = Input_Alignment
        self.Results = Results
        self.Length = Length
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp
        self.true_tree = true_tree
        try:
            os.rmdir('%s' % self.Temp)
        except OSError:
            pass
        self.checkpoint = checkpoint


    def stage1(self, Min_lambda, Max_lambda, Width_lambda, Iterations, Proposals):

        # get starting value
        current_lambda = random.uniform(Min_lambda, Max_lambda)

        # initiate trainer and a proposer
        Stage1_Trainer = Simple_Training(IQ_Tree = self.IQ_Tree, Model = self.Model,
            Chunks = self.Chunks, Length = self.Length,
            Input_Alignment = self.Input_Alignment, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp)
        My_proposer = Proposer()

        # initiate current distance and lists for storing results
        dist_curr = float('inf')
        current_lambdas = []
        current_distances = []

        # do training
        for i in range(Iterations):
            print("ITER", i)
            print("time", datetime.datetime.now().time())

            # calculate temperature
            T = utils.temperature(i, Iterations)

            # propose changes
            proposed_lambda_best = None
            proposed_dist_best = float('inf')

            for rep in range(0, Proposals):

                # propose a new lambda
                proposed_lambda = My_proposer.proposal(curr_value = current_lambda,
                    multiplier = T, proposal_width = Width_lambda,
                    proposal_min = Min_lambda, proposal_max = Max_lambda)

                # calculate the distance between the average proportions of invariant sites
                proposed_dist = Stage1_Trainer.simple_loss_stage1(proposed_lambda)

                if proposed_dist < proposed_dist_best:
                    print("keeping proposal", proposed_lambda, proposed_dist)
                    proposed_dist_best = proposed_dist
                    proposed_lambda_best = proposed_lambda

            if proposed_dist_best <= dist_curr:
                p_accept = 1
            else:
                p_accept =  dist_curr/proposed_dist_best*T

            rand = np.random.rand()
            accept = rand < p_accept

            # if accept, retrain
            if accept:
                print("Accepted lambda ", proposed_lambda_best, 'because ', p_accept)
                dist_curr = proposed_dist_best
                current_lambda = proposed_lambda_best

            current_lambdas.append(current_lambda)
            current_distances.append(dist_curr)
            print('Current distance: ', dist_curr, " Current lambda: ", current_lambda, '\n')

        utils.write_results_stage1(self.Results, current_lambdas, dist_curr)
        return (current_lambda)


    def stage2(self, Pretraining_Iterations, Pretraining_Epochs, Training_Iterations, Training_Epochs, MaxTrees, inferred_lambda, start_tree, start_iter, checkpoint_interval):

        
        
        
        the_discriminator = discriminator()
        
        if self.checkpoint == True:
            the_discriminator.load_weights('./%s/Checkpoint_model' % self.Results)
            discriminator_state = the_discriminator
            
            # get Max SNPs
            MaxSNPs = int(open(self.Results + '/Checkpoint_MaxSNPs.txt', 'r').readlines()[-1].strip())

        else:
            # calculate max SNPs
            MaxSNPs = utils.calc_max_snps(self.Input_Alignment, self.Length, self.Chunks)


        # get Stage2_Trainer
        Stage2_Trainer = Training(discriminator = the_discriminator, IQ_Tree = self.IQ_Tree,
            Model = self.Model, Chunks = self.Chunks, Length = self.Length,
            Input_Alignment = self.Input_Alignment, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp,
            birthRate = inferred_lambda, numPretrainingEpochs = Pretraining_Epochs,
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
    
            current_tree = best_tree
            current_loss = float('inf')
            current_accuracy = 0
    
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
                mover = treemoves.TreeMove(current_tree, T, MaxTrees, inferred_lambda)
                trees = mover()
                #trees.append(current_tree)

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


    def stage2_generator(self, Pretraining_Iterations, Pretraining_Epochs, Training_Iterations, Training_Epochs, MaxTrees, inferred_lambda):
        """this function is only for testing the generator."""
        print('Generator Run: Begin training.')

        current_tree = utils.get_start_tree(start_tree='Random', birthRate = inferred_lambda,
            IQ_Tree = self.IQ_Tree, Model = self.Model, Input_Alignment = self.Input_Alignment,
            Input_Order = self.Input_Order, Input_Match_dict = self.Input_Match_dict,
            Temp = self.Temp, Results = self.Results)
        current_distance = float('inf')

        # lists to store results
        results_trees = []
        distances = []

        # get ref tree
        try:
            ref_tree = ete3.Tree(self.true_tree)
        except:
            sys.exit('ERROR: You must provide a true tree to use the generator-only inference mode.')

        for i in range(Training_Iterations):
             print('\nCurrent distance: ', current_distance, " Current tree: ", current_tree)
             print("ITER", i)
             print("time", datetime.datetime.now().time())

             # calculate temperature
             T = utils.temperature(i, Training_Iterations)

             # if i == 0, get start trees
             if i == 0:
                 trees = [current_tree]

             else:
                 mover = treemoves.TreeMove(current_tree, T, MaxTrees, inferred_lambda)
                 trees = mover()
             proposed_distance_best = float('inf')
             proposed_tree_best = None

             for proposed_tree in trees:

                 # calculate the distance
                 comp_tree = ete3.Tree(proposed_tree)
                 comparison = comp_tree.compare(ref_tree, unrooted = True)
                 proposed_distance = comparison['norm_rf']

                 if proposed_distance < proposed_distance_best:
                     proposed_distance_best = proposed_distance
                     proposed_tree_best = proposed_tree
                     print('Keep tree because distance ', proposed_distance)

             if proposed_distance_best < current_distance:
                 p_accept = 1

             else:
                 #p_accept =  (current_distance+1e-9)/(proposed_distance_best+1e-9)*T
                 p_accept =  (current_distance+1e-9)/(proposed_distance_best+1e-9)

             rand = np.random.rand()
             accept = rand < p_accept


             # if accept, record
             if accept:
                 print("Accepted tree ", proposed_tree_best, ' because ', p_accept)
                 current_distance = proposed_distance_best
                 current_tree = proposed_tree_best

             # add info to lists
             results_trees.append(current_tree)
             distances.append(current_distance)

        utils.write_results_stage2_generator(results_trees, distances, self.Results)

    def stage2_discriminator(self, Pretraining_Iterations, Pretraining_Epochs, Training_Iterations, Training_Epochs, MaxTrees, inferred_lambda, start_tree):
        """this function is only for testing the discriminator."""

        # calculate max SNPs
        MaxSNPs = utils.calc_max_snps(self.Input_Alignment, self.Length, self.Chunks)

        # pretraining
        print('Begin pre-training.\n')

        # get Stage2_Trainer
        Stage2_Trainer = Training(discriminator = discriminator(), IQ_Tree = self.IQ_Tree,
            Model = self.Model, Chunks = self.Chunks, Length = self.Length,
            Input_Alignment = self.Input_Alignment, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp,
            birthRate = inferred_lambda, numPretrainingEpochs = Pretraining_Epochs,
            numTrainingEpochs = Training_Epochs, maxSNPs = MaxSNPs)

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
        print(best_tree)


        # set up automated walk
        mover = treemoves.TreeMove(self.true_tree, None, 1, inferred_lambda)
        trees_to_visit = mover.autowalk(Training_Iterations)

        print('Begin training.')

        current_tree = best_tree
        current_loss = float('inf')
        current_accuracy = 0

        # lists to store results
        results_trees = []
        generator_loss = []
        generator_fake_acc = []
        discriminator_loss = []
        discriminator_real_acc = []
        discriminator_fake_acc = []

        for i in range(Training_Iterations):
            print('\nCurrent loss: ', current_loss, " Current tree: ", current_tree)
            print("ITER", i)
            print("time", datetime.datetime.now().time())

            # calculate temperature
            T = utils.temperature(i, Training_Iterations)

            proposed_tree = trees_to_visit[i]



            # calculate the generator loss
            proposed_loss, proposed_accuracy = Stage2_Trainer.generator_loss_stage2(proposed_tree)

            if proposed_loss < current_loss:
                p_accept = 1

            else:
                p_accept =  current_loss/proposed_loss*T*0.1
                #p_accept =  current_loss/proposed_loss

            rand = np.random.rand()
            accept = rand < p_accept


            # if accept, retrain
            if accept:
                print("Accepted tree ", proposed_tree, ' because ', p_accept)
                current_loss = proposed_loss
                current_accuracy = proposed_accuracy
                current_tree = proposed_tree


                # train
                real_acc, fake_acc, disc_loss = Stage2_Trainer.train_sa_stage2(proposed_tree)

            # add info to lists
            results_trees.append(current_tree)
            generator_loss.append(current_loss)
            generator_fake_acc.append(current_accuracy)
            discriminator_loss.append(float(disc_loss))
            discriminator_real_acc.append(real_acc)
            discriminator_fake_acc.append(fake_acc)

        # write results
        utils.write_results_stage2(results_trees, generator_loss, generator_fake_acc, discriminator_loss, discriminator_real_acc, discriminator_fake_acc, self.Results, self.true_tree)


