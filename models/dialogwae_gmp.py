import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar
from modules import Encoder, ContextEncoder, MixVariation, Decoder, MixVariationFixCompo   
from .dialogwae import DialogWAE


class DialogWAE_GMP(DialogWAE):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DialogWAE_GMP, self).__init__(config, vocab_size, PAD_token)
        self.n_components = config['n_prior_components']
        self.gumbel_temp = config['gumbel_temp']
        self.prior_net = MixVariation(config['n_hidden'], config['z_size'], self.n_components, self.gumbel_temp) # p(e|c)
        
    def sample_code_prior_fix(self, c):
        # define prior net which only select a specific component
        self.fix_prior_net = MixVariationFixCompo(self.prior_net.input_size, self.prior_net.z_size, self.prior_net.n_components, self.prior_net.gumbel_temp, select_compo=2)
        
        # replace and copy weights
        for i in range(len(self.prior_net.pi_net)):
            if type(self.prior_net.pi_net[i]) == torch.nn.modules.activation.Tanh:
                continue
            self.fix_prior_net.pi_net[i].weight.data = self.prior_net.pi_net[i].weight.data.clone()

        for i in range(len(self.prior_net.fc)):
            if type(self.prior_net.fc[i]) == torch.nn.modules.activation.Tanh:
                continue
            self.fix_prior_net.fc[i].weight.data = self.prior_net.fc[i].weight.data.clone()

        self.fix_prior_net.context_to_mu.weight.data = self.prior_net.context_to_mu.weight.data.clone()
        self.fix_prior_net.context_to_logsigma.weight.data = self.prior_net.context_to_logsigma.weight.data.clone()
        self.fix_prior_net.to(c.device)
        
        e, _, _ = self.fix_prior_net(c)
        z = self.prior_generator(e)
        return z 
        
    def sample_fix(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):    
        self.context_encoder.eval()
        self.decoder.eval()

        # encode context into embedding
        c = self.context_encoder(context, context_lens, utt_lens, floors) 
        c_repeated = c.expand(repeat, -1)
        prior_z = self.sample_code_prior_fix(c_repeated) 

        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy") 
        return sample_words, sample_lens 

    
    


           
   
    

    



