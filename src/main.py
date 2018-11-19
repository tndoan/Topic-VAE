import torch, math
import util
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

class TopicVAE(nn.Module):
    def __init__(self, K, emb_size, emb_vector, hidden_size):
        super(TopicVAE, self).__init__()
        self.K = K # number of topics
        self.emb_vector = emb_vector
        self.hidden_size = hidden_size

        # some layers for the model
        self.RNNC = nn.RNNCell(emb_size, K, 1)
        # define some layers in Eq 6
        self.W_y = nn.Linear(emb_size, hidden_size) 
        self.W_u = nn.Linear(hidden_size, 100)
        self.W_s = nn.Linear(hidden_size, 100)
        self.W_k = []
        for i in xrange(K):
            self.W_k.append(nn.Linear(hidden_size, 1))

        # initialize
        self.alpha = np.ones(K) / K # init alpha

        # approximate Dirichlet distribution by logistic normal distribution
        self.mu_1    = np.log(self.alpha) - np.sum(self.alpha) / K 
        inv_alpha    = 1.0 / self.alpha
        self.sigma_1 = (1.0 - 2.0 / K) * inv_alpha + np.sum(inv_alpha) / (K**2) 

    def find_h_of_sentence(self, sentence):
        # implementation of Eq 3
        h = torch.randn(1, self.hidden_size)
        h_list = list()
        for (wId, isSW) in sentence: # (word id, is stopword)
            # encode h 
            emb = self.emb_vector.get(wId) # embedding vector of word
            if not isSW:
                h_next = self.RNNC(emb, h)
                h = h_next + h
            h_list.append(h)
        q_z = list()
        for k in xrange(self.K):
            q_z.append(F.softmax(self.W_k[k](h)))
        return h_list, q_z

    def encode(self, sentence):
        h_list, q_z = self.find_h_of_sentence(sentence)
        emb = torch.zeros(emb_size) 
        for (wId, isSW) in sentence:
            emb = emb + self.emb_vector.get(wId)
        # encode by deep learning. It could be modified later
        gamma_d = F.relu(self.W_y(emb)) # in eq 6, it is tanh
        mu_0 = self.W_u(gamma_d)
        log_sigma_0 = self.W_s(gamma_d)
        return mu_0, log_sigma_0, h_list, q_z

    def reparameterize(self, mu_0, log_sigma_0):
        std = torch.exp(0.5 * log_sigma_0)
        exp = torch.randn_like(std)
        return exp.mul(std).add_(mu_0)

    def decode(self, z): 
        # output: theta_d in Eq 5
        return torch.sigmoid(z) # we could let z go through some neural layers

    def forward(self, sentence):
        # input: sentence is the sentence of one document
        # ouput: theta, mu, log_sigma
        mu, log_sigma, h_list, q_z = self.encode(sentence)
        z = self.reparameterize(mu, log_sigma)
        return self.decode(z), mu, log_sigma, h_list, q_z

def loadEmb():
    return dict() # TODO

data = util.readData('../data/featureTest.txt')
A = dict()
phi = torch.randn(D, D, requires_grad=True)
emb_vector = loadEmb() # TODO

def loss_func(K, sigma_1, sigma_0, mu_1, mu_0, theta, h_list, q_z, sentence):
    # loss function for one document
    # Eq 8
    inv_sigma_1 = torch.inverse(torch.diag(sigma_1))
    mu_1_2 = mu_1 - mu_0

    # 1st
    result = 0.5 * (torch.trace(torch.mul(inv_sigma_1, sigma_0)) + \
            torch.mul(torch.mul(torch.t(mu_1_2), inv_sigma_1), mu_1_2) - K +\
            math.log(float(torch.prod(sigma_1))/ float(torch.prod(sigma_0))))
    
    # 2nd
    result += torch.sum(theta) - K * math.log(torch.sum([math.exp(i) for i in theta]))

    for i in xrange(len(q_z)):
        q = q_z[i]
        # 3rd
        result += - q * math.log(q)
        # 4th part
        for j in xrange(len(sentence)):
            (wId, isSW) = sentence[j]
            wVector = A.get(wId)
            if wVector == None:
                wVector = torch.randn(1,  hidden_size, requires_grad=True)
            temp = torch.dot(wVector, h)
            if not isSW:
                temp += torch.dot(emb_vector.get(wId), phi[i])
            result += q * math.log(temp)

    return result

device = torch.device("cuda" if args.cuda else "cpu")
model = TopicVAE(10).to(device)
optimizer = optim.Adam([model.parameters(), A, phi], lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for doc in data:
        for sentence in doc:
            decode_z, mu, log_sigma, h_list, q_z = model(sentence)
            loss = loss_func(K, model.sigma_1, log_sigma, model.mu_1, mu, decode_z, h_list, q_z, sentence)
            
            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train(10)
