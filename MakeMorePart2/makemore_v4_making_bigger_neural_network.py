
import torch
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

print(words[:8])
print("len of words", len(words))

# build the vocabulary of characters and mapping to/from integers

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}
print("itos", itos)

#========================================>
print(" =============================== Line 27,, building the dataset , chcecking out on 1 word/example =============================== ")
block_size = 3 # context length: how many characters do we take to predict the next one?

X,Y = [], []  # X==> input to the neural net; Y ==> are the labels


length_taken = len(words)
# example of first 5 words
for w in words[:length_taken]:
    
    #print("Line 35: word", w)
    context = [0] * block_size

    for ch in w + '.':
        
        ix = stoi[ch]
        X.append(context)
        #print("X", X)
        Y.append(ix)
        
# =============================================================================
#         ... -----> e
#         ..e -----> m
#         .em -----> m
#         emm -----> a
#         mma -----> .
# =============================================================================
        #print("Line 45",''.join(itos[i] for i in context), '----->', itos[ix])
        
        context = context[1:] + [ix]   # crop and append  # rolling window of context
        #print("Line 55, context", context)  # context [0, 0, 5] ==> ..e ; context [5, 13, 13] ==> emm
        
# =============================================================================
# for word emma: (create 5 inputs), 1 output
#
# X tensor([[ 0,  0,  0],
#         [ 0,  0,  5],
#         [ 0,  5, 13],
#         [ 5, 13, 13],
#         [13, 13,  1]])
#
#
# Y tensor([ 5, 13, 13,  1,  0])
# =============================================================================

X = torch.tensor(X)
Y = torch.tensor(Y)

print("X", X)
print("Y",Y)

print("Line 76: X.shape, Y.shape", X.shape, Y.shape)

#========================================================>
#building a look-up table.. 
# 27 characters (26 characters and one dot) and embed them into lower dimension space (say 2 dimensional space)
# inpaper there were 17,000 words which got embedded into 30 dimensional space. 

# each one of 27 can get embedded into 2 d

C = torch.randn((27,2))


# will not use this one, it is slower.. directly will be using C (embedding table)
# =============================================================================
# # creating input for neural network
# x_one_hot_encoded =  F.one_hot(torch.tensor(length_taken), num_classes = 27).float()
# 
# x_one_hot_encoded @ C
# =============================================================================

# how to embed (5,3) integers (5 examples from 1 word, 3 is the context length/block size) ==> take C[X]
#print("Line 99: C[X], C[X].shape",C[X], C[X].shape)

emb = C[X]
print("Line 102 emb.shape", emb.shape)  #shape ==>  torch.Size([5, 3, 2])

W1 = torch.randn((6,100))   # input layer (3 *2) ==> 3 context length, 2 embedding size. ; output is 27 or can be variable also

b1 = torch.randn(100)

# as an input embedding as to be multiplied with W1, how can we do that? 5 examples are stacked in embedding


emb[:,0,:].shape  # plucks out the embedding for the 1st context vectors in all 5 examples ;torch.Size([5, 2])
emb[:,1,:].shape  # plucks out the embedding for the 2nd context vectors in all 5 examples ;torch.Size([5, 2])
emb[:,2,:].shape  # plucks out the embedding for the 3rd context vectors in all 5 examples ;torch.Size([5, 2])

# we have to make a sequence out of them by concatenating them column wise (3 columns)
# block size = 3, not generalizable.. 
concatenated = torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], 1)

print("concatenated.shape", concatenated.shape)  #torch.Size([5, 6])

concatenated_beautiful_way = torch.cat(torch.unbind(emb,1),1)   # this is very inefficient.. 
print("concatenated_beautiful_way.shape", concatenated_beautiful_way.shape)  #torch.Size([5, 6])

#=====================>
# much better way  .view is extremely efficient

a = torch.arange(18)
print("line 127 a", a)

print("line 129: a.view(2,9)", a.view(2,9))
print("line 129: a.view(3,3,2)", a.view(3,3,2))

#======================================>
# =============================================================================
# print("emb.view(5,6) == torch.cat(torch.unbind(emb,1),1)",
#       emb.view(5,6) == torch.cat(torch.unbind(emb,1),1))
# =============================================================================

concatenated_beautiful_way_2 = emb.view(emb.shape[0],6)  # emb.shape[0] is 5
#or concatenated_beautiful_way_2 = emb.view(-1,6)   # pytorch derive the dimension that it is 5

hidden =  concatenated_beautiful_way_2 @ W1 + b1

print("Line 139, hidden.shape", hidden.shape)  # these will be loke logits

hidden = torch.tanh(hidden)    # hidden layer of activation # for each of the 5 inputs.. 
print("Line 144, hidden.shape", hidden.shape)

# ==============>

# creating the final layer
# 27 characters can be the outptu
W2 = torch.randn((100, 27))  

b2 = torch.randn(27)

logits = hidden @ W2 + b2


print("Line 157", logits.shape)
# ============>
# similar to part 1:
    
# exponentiate the logits to get the count. 
counts = logits.exp()

# convert count to probability
prob = counts/counts.sum(1, keepdims = True)

print("Line 168: prob.shape", prob.shape)

#=============================================>
# index into 5 rows of prob, from each row of prob, we want to grab the probability of Y
# that should be the answer
 
# [torch.arange(emb.shape[0]), Y]  ==> [tensor([0, 1, 2, 3, 4]), tensor([ 5, 13, 13,  1,  0])]

# now apply prob on ==>  prob[tensor([0, 1, 2, 3, 4]), tensor([ 5, 13, 13,  1,  0])]

# gives the output as tensor([1.3635e-08, 6.5415e-04, 9.1457e-06, 4.5404e-11, 4.6578e-06])

print(prob[torch.arange(emb.shape[0]), Y])

# get those probabilities
probabilities = prob[torch.arange(emb.shape[0]), Y]

# get the log probabilities
log_probabilities = prob[torch.arange(emb.shape[0]), Y].log()

# get the average log probabilities
average_log_probabilities = prob[torch.arange(emb.shape[0]), Y].log().mean()

# create negative log likelihood loss

negative_log_likelihood_loss = - average_log_probabilities

print("Line 195, negative_log_likelihood_loss", negative_log_likelihood_loss)

#====================================================================>
print("============================== Line 198, making everything respectable, rewriting in short and use of generator==============================")

print("Line 200: X.shape, Y.shape", X.shape, Y.shape)

g = torch.Generator().manual_seed(2147483648)  # for repreoducibility
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,100), generator = g)
b1 = torch.randn(100, generator = g)

W2 = torch.randn((100,27), generator = g)
b2 = torch.randn(27, generator = g)

parameters = [C,W1,b1,W2,b2]

print("checking number of parameters", sum(p.nelement() for p in parameters))  # number of parameters in total

emb = C[X] #(5,3,2)
hidden1 = emb.view(-1,6)
hidden = torch.tanh(hidden1 @ W1 + b1) 

logits = hidden @ W2 + b2 #(5, 27)

counts = logits.exp()
prob = counts/ counts.sum(1, keepdims = True)
loss =  -prob[torch.arange(emb.shape[0]), Y].log().mean()
print("Line 222, loss", loss)


#=======================================>
# =============================================================================
# we can use cross entropy instead of manuall y calculating:
#     counts = logits.exp()
#     prob = counts/ counts.sum(1, keepdims = True)
#     loss =  -prob[torch.arange(emb.shape[0]), Y].log().mean()
# =============================================================================

loss_using_cross_entropy = F.cross_entropy(logits, Y)
print("Line 234, loss_using_cross_entropy", loss_using_cross_entropy)

#===========================================================================================>
print(" ============================ Line 240, rewriting ====================================")

print("Line 240: X.shape, Y.shape", X.shape, Y.shape)

g = torch.Generator().manual_seed(2147483648)  # for repreoducibility
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,100), generator = g)
b1 = torch.randn(100, generator = g)

W2 = torch.randn((100,27), generator = g)
b2 = torch.randn(27, generator = g)

parameters = [C,W1,b1,W2,b2]

print("checking number of parameters", sum(p.nelement() for p in parameters))  # number of parameters in total

for p in parameters:
    p.requires_grad = True
    
# on all words not on mini batch
for _ in range(1):
    # forward pass
    emb = C[X] #(5,3,2)
    hidden1 = emb.view(-1,6)
    hidden = torch.tanh(hidden1 @ W1 + b1) 
    
    logits = hidden @ W2 + b2 #(5, 27)
    loss_using_cross_entropy = F.cross_entropy(logits, Y)
    #print("Line 265, loss_using_cross_entropy", loss_using_cross_entropy.item())
    
    # backward pass
    for p in parameters:
        p.grad = None
    
    loss_using_cross_entropy.backward()
    
    # parameter update
    for p in parameters:
        
        p.data += -0.1*p.grad
        
# even training loss cannot go to zero.. becase of the example we are using.. 
# Line 45 ... -----> e
# Line 45 ... -----> o 
# see that dot have to predict, both e and o, and so many others like it. .
# which is difficult.. 
    
print("Line 284, loss_using_cross_entropy", loss_using_cross_entropy.item())

#==================================================================================
# we have used all the examples, and forwarding and backwarding through all of them.. 
# lets do forwarding and backwarding through mini batch instead. 

# we will make forward, backward, update on mini batch.. 
# mini batch chosen randomly
print(" =============================== line 294, training on mini batch now and setting up variable learning rate ====================")


g = torch.Generator().manual_seed(2147483648)  # for repreoducibility
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,100), generator = g)
b1 = torch.randn(100, generator = g)

W2 = torch.randn((100,27), generator = g)
b2 = torch.randn(27, generator = g)

parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True
    
lre = torch.linspace(-3,0, 1000)
lrs = 10**lre

lri = []
lossi = []
for i in range(1000):
    
    # let's construct a minibatch of size 32 (chosen randomly)
    # we have to choose 32 indices from X and Y both, for a minibatch
    
    ix = torch.randint(0, X.shape[0], (32,))
    
    # forward pass
    mini_batch_X = X[ix]
    mini_batch_Y = Y[ix]

    emb = C[mini_batch_X] #(32,3,2)   # ==> 32 because we are indexing into X for 32 examples, from minibatch
    
    hidden1 = emb.view(-1,6)
    hidden = torch.tanh(hidden1 @ W1 + b1) 
    
    logits = hidden @ W2 + b2 #(32, 27)
    
    loss_using_cross_entropy = F.cross_entropy(logits,mini_batch_Y)
    
    #print("Line 265, loss_using_cross_entropy", loss_using_cross_entropy.item())
    
    # backward pass
    for p in parameters:
        p.grad = None
    
    loss_using_cross_entropy.backward()
    
    # parameter update
    lr  = lrs[i]
    for p in parameters:
        
        # we guessed 0.1, we do not know if we are stepping too slow or too fast. 
        p.data += -lr*p.grad

    lri.append(lr)
    lossi.append(loss_using_cross_entropy.item())
    
print("Line 343, loss_using_cross_entropy", loss_using_cross_entropy.item())

plt.plot(lri, lossi)

#==========================================================================>
# around 0.1 did the best in terms of learning rate as can be seen from the plot.. 
#so just using that

print(" =============================== line 362, training on mini batch now and constanat learning rate ====================")


g = torch.Generator().manual_seed(2147483648)  # for repreoducibility
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,100), generator = g)
b1 = torch.randn(100, generator = g)

W2 = torch.randn((100,27), generator = g)
b2 = torch.randn(27, generator = g)

parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True
    
lre = torch.linspace(-3,0, 1000)
lrs = 10**lre

lri = []
lossi = []
for i in range(1000):
    
    # let's construct a minibatch of size 32 (chosen randomly)
    # we have to choose 32 indices from X and Y both, for a minibatch
    
    ix = torch.randint(0, X.shape[0], (32,))
    
    # forward pass
    mini_batch_X = X[ix]
    mini_batch_Y = Y[ix]

    emb = C[mini_batch_X] #(32,3,2)   # ==> 32 because we are indexing into X for 32 examples, from minibatch
    
    hidden1 = emb.view(-1,6)
    hidden = torch.tanh(hidden1 @ W1 + b1) 
    
    logits = hidden @ W2 + b2 #(32, 27)
    
    loss_using_cross_entropy = F.cross_entropy(logits,mini_batch_Y)
    
    #print("Line 265, loss_using_cross_entropy", loss_using_cross_entropy.item())
    
    # backward pass
    for p in parameters:
        p.grad = None
    
    loss_using_cross_entropy.backward()
    
    # parameter update
    lr  = 0.01
    for p in parameters:
        
        # we guessed 0.1, we do not know if we are stepping too slow or too fast. 
        p.data += -lr*p.grad

    #lri.append(lr)
    #lossi.append(loss_using_cross_entropy.item())

print("Line 421, loss_using_cross_entropy", loss_using_cross_entropy.item())
    
#print("Line 343, loss_using_cross_entropy", loss_using_cross_entropy.item())

#plt.plot(lri, lossi)
#=================:
    #%% MAKING NEURAL NETWORK BIGGER
print(" =========== Line 427: Making Neural network bigger =======")


def build_dataset(words):
    block_size = 3 # context length: how many characters do we take to predict the next one?

    X,Y = [], []  # X==> input to the neural net; Y ==> are the labels


    length_taken = len(words)
    
    for w in words[:length_taken]:
        

        context = [0] * block_size

        for ch in w + '.':
            
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)

            
            context = context[1:] + [ix]   # crop and append  # rolling window of context

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    
    return X,Y

import random
random.seed(42)
random.shuffle(words)  # shuffing the words
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print("Xte.shape, Yte.shape",Xte.shape, Yte.shape) #Xte.shape, Yte.shape torch.Size([22866, 3]) torch.Size([22866])



g = torch.Generator().manual_seed(2147483648)  # for repreoducibility
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,300), generator = g)
b1 = torch.randn(300, generator = g)

W2 = torch.randn((300,27), generator = g)
b2 = torch.randn(27, generator = g)

parameters = [C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad = True
    
lre = torch.linspace(-3,0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []
for i in range(10000):
    
    # let's construct a minibatch of size 32 (chosen randomly)
    # we have to choose 32 indices from X and Y both, for a minibatch
    
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # forward pass
    mini_batch_X = Xtr[ix]
    mini_batch_Y = Ytr[ix]

    emb = C[mini_batch_X] #(32,3,2)   # ==> 32 because we are indexing into X for 32 examples, from minibatch
    
    hidden1 = emb.view(-1,6)
    hidden = torch.tanh(hidden1 @ W1 + b1) 
    
    logits = hidden @ W2 + b2 #(32, 27)
    
    loss_using_cross_entropy = F.cross_entropy(logits,mini_batch_Y)
    
    #print("Line 265, loss_using_cross_entropy", loss_using_cross_entropy.item())
    
    # backward pass
    for p in parameters:
        p.grad = None
    
    loss_using_cross_entropy.backward()
    
    # parameter update
    lr  = 0.1
    for p in parameters:
        
        # we guessed 0.1, we do not know if we are stepping too slow or too fast. 
        p.data += -lr*p.grad

    #lri.append(lr)
    stepi.append(i)
    lossi.append(loss_using_cross_entropy.item())

print("Line 528, Training loss_using_cross_entropy", loss_using_cross_entropy.item())

# evaluate the loss on Xdev and Ydev
emb = C[Xdev]
hidden1 = emb.view(-1,6)
hidden = torch.tanh(hidden1 @ W1 + b1) 
logits = hidden @ W2 + b2 #(32, 27)
loss_using_cross_entropy = F.cross_entropy(logits,Ydev)
plt.plot(stepi, lossi)

print("Line 537, Dev loss_using_cross_entropy", loss_using_cross_entropy.item())

#============>
# there is a lot of noise in the steps (it is thick).. one reason might be that batch size is quite low for trianing.. 

# visualize the embeddings trained by the neural network.. 
plt.figure(figsize = (8,8))
plt.scatter(C[:,0].data, C[:,1].data, s = 200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha = "center", va = "center", color = "white")

plt.grid('minor')


