"""Parts of this code are adapted from Wang et al., 2021, Molecular Ecology Resources, https://doi.org/10.1111/1755-0998.13386."""
from tensorflow.keras import losses, optimizers
from tensorflow import GradientTape, ones_like, zeros_like, math
import numpy as np
from simulators import *


class Simple_Training(object):

    def __init__(self, IQ_Tree, Model, Chunks, Length, Input_Alignment, Input_Order, Input_Match_dict, Temp):
        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Length = Length
        self.Input_Alignment = Input_Alignment
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp
        self.Simulator = Simulator(IQ_Tree = self.IQ_Tree, Model = self.Model,
            Chunks = self.Chunks, Length = self.Length,
            Input_Alignment = self.Input_Alignment, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp)

    def simple_loss_stage1(self, current_lambda):
        """Simulate data and calculate generator loss."""

        ## simulate data
        simulated_data = self.Simulator.simulateLambda(current_lambda)

        # subset N chunks of empirical data
        empirical_data = self.Simulator.countPinvChunk()

        # calculate averages to compute difference
        avg_empirical = np.average(empirical_data)
        avg_simulated = np.average(simulated_data)

        return abs(avg_empirical - avg_simulated)


class Training(object):

    def __init__(self, discriminator, IQ_Tree, Model, Chunks, Length, Input_Alignment, Input_Order, Input_Match_dict, Temp, birthRate, numPretrainingEpochs, numTrainingEpochs, maxSNPs):
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
                decay_steps=10000,
                    decay_rate=0.9)
        self.disc_optimizer = optimizers.Adam()
        self.discriminator = discriminator
        self.IQ_Tree = IQ_Tree
        self.Model = Model
        self.Chunks = Chunks
        self.Length = Length
        self.Input_Alignment = Input_Alignment
        self.Input_Order = Input_Order
        self.Input_Match_dict = Input_Match_dict
        self.Temp = Temp
        self.birthRate = birthRate
        self.numPretrainingEpochs = numPretrainingEpochs
        self.numTrainingEpochs = numTrainingEpochs
        self.maxSNPs = maxSNPs
        self.Simulator = Simulator(IQ_Tree = self.IQ_Tree, Model = self.Model,
            Chunks = self.Chunks, Length = self.Length,
            Input_Alignment = self.Input_Alignment, Input_Order = self.Input_Order,
            Input_Match_dict = self.Input_Match_dict, Temp = self.Temp)


    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss."""

        # accuracy
        real_acc = np.sum(real_output >=0) # positive logit => pred 1
        fake_acc = np.sum(fake_output <0) # negative logit => pred 0

        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # add on entropy
        real_entropy = self.cross_entropy(real_output, real_output)
        fake_entropy = self.cross_entropy(fake_output, fake_output)
        entropy = math.scalar_mul(0.001/2, math.add(real_entropy, fake_entropy))
        total_loss -= entropy

        return total_loss, real_acc, fake_acc


    def train_step_stage2(self, real_regions, real_pinv, all_generated_regions, all_generated_pinv):
        """One mini-batch for the discriminator."""

        with GradientTape() as disc_tape:


            # feed real and generated regions to discriminator
            real_output = self.discriminator.call_mathieson(real_regions,training=True)
            fake_output = self.discriminator.call_mathieson(all_generated_regions, training=True)



            # calculate discirminator loss
            disc_loss, real_acc, fake_acc = self.discriminator_loss(real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss



    def pretraining_stage2(self, ):
        """Simulate data and calculate generator loss."""

        # Simulate a tree
        tree = self.Simulator.simulateLambdaTree(self.birthRate)

        for epoch in range(self.numPretrainingEpochs):

            # simulate data
            all_generated_regions, all_generated_pinv = self.Simulator.simulateonTree(tree, self.maxSNPs, self.birthRate)

            # subset N chunks of empirical data
            all_empirical_regions, all_empirical_pinv = utils.sequencetonparrayChunk(align = self.Input_Alignment,
                numTaxa = len(self.Input_Order), maxSNPs = self.maxSNPs, Length = self.Length, Chunks = self.Chunks)
            # do TRAINING
            real_acc, fake_acc, disc_loss = self.train_step_stage2(all_empirical_regions, all_empirical_pinv, all_generated_regions, all_generated_pinv)

            # print results
            if (epoch+1) % 5 == 0:
               template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
               print(template.format(epoch+1,
                   disc_loss,
                   real_acc/self.Chunks * 100,
                   fake_acc/self.Chunks * 100))



        return real_acc/self.Chunks, fake_acc/self.Chunks, float(disc_loss), tree

    def generator_loss_stage2(self, proposed_tree):
        #proposed_tree, Model, Num_alignments, Length, empirical_sporder, maxSNPs, Num_taxa, Iq_tree):
       # simulate the fake data

        all_generated_regions = []
        all_generated_pinv = []

        # simulate data
        all_generated_regions, all_generated_pinv = self.Simulator.simulateonTree(proposed_tree, self.maxSNPs, self.birthRate)


        all_generated_regions = np.array(all_generated_regions)
        all_generated_pinv = np.array(all_generated_pinv)
        fake_output = self.discriminator.call_mathieson(all_generated_regions,training=False)

        fake_acc = np.sum(fake_output <0)/self.Chunks # negative logit => pred 0
        loss = self.cross_entropy(ones_like(fake_output), fake_output)

        return loss.numpy(), fake_acc


    def train_sa_stage2(self, proposed_tree):


        for epoch in range(self.numTrainingEpochs):

           # simulate data
           all_generated_regions, all_generated_pinv = self.Simulator.simulateonTree(proposed_tree, self.maxSNPs, self.birthRate)

           all_generated_regions = np.array(all_generated_regions)
           all_generated_pinv = np.array(all_generated_pinv)

           # subset N chunks of empirical data
           all_empirical_regions, all_empirical_pinv = utils.sequencetonparrayChunk(align = self.Input_Alignment,
                numTaxa = len(self.Input_Order), maxSNPs = self.maxSNPs, Length = self.Length, Chunks = self.Chunks)

           # do training
           real_acc, fake_acc, disc_loss = self.train_step_stage2(all_empirical_regions, all_empirical_pinv, all_generated_regions, all_generated_pinv)


           # print results
           if (epoch+1) % 5 == 0:
               template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
               print(template.format(epoch+1,
                   disc_loss,
                   real_acc/self.Chunks * 100,
                   fake_acc/self.Chunks * 100))

       

        return real_acc/self.Chunks, fake_acc/self.Chunks, disc_loss, self.discriminator

