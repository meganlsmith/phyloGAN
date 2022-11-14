from tensorflow import GradientTape, ones_like, zeros_like, math
from tensorflow.keras import losses, optimizers
import numpy as np
import os


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



class Training(object):

    def __init__(self, discriminator, iqtree_path, maxSNPs, Chunks):
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
                decay_steps=10000,
                    decay_rate=0.9)
        self.disc_optimizer = optimizers.Adam()
        self.discriminator = discriminator
        self.iqtree_path = iqtree_path
        self.maxSNPs = maxSNPs
        self.Chunks = Chunks


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
            real_output = self.discriminator.call_mathieson(real_regions, training=True)
            fake_output = self.discriminator.call_mathieson(all_generated_regions, training=True)



            # calculate discirminator loss
            disc_loss, real_acc, fake_acc = self.discriminator_loss(real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return real_acc, fake_acc, float(disc_loss)


    def train_sa_stage2(self, epochs, generated_regions, generated_var, empirical_regions, empirical_var):
    
    
        for epoch in range(epochs):
    
           # do training
           real_acc, fake_acc, disc_loss = self.train_step_stage2(empirical_regions, empirical_var, generated_regions, generated_var)
    
    
           # print results
           if (epoch+1) % 5 == 0:
               template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
               print(template.format(epoch+1,
                   disc_loss,
                   real_acc/self.Chunks * 100,
                   fake_acc/self.Chunks * 100))
    
        # remove the tree
        #
        return real_acc/self.Chunks, fake_acc/self.Chunks, float(disc_loss), self.discriminator


    def generator_loss(self, generated_regions, generated_var):


        fake_output = self.discriminator.call_mathieson(generated_regions, training=False)

        fake_acc = np.sum(fake_output <0)/self.Chunks # negative logit => pred 0
        loss = self.cross_entropy(ones_like(fake_output), fake_output)

        return loss.numpy(), fake_acc
