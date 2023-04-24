
import torch
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

import torch.nn.functional as F

#%matplotlib inline

"""
Character Models, practice of andrej Karpathy MakeMore series

Current implementation follows a few key papers:

Bigram (one character predicts the next one with a lookup table of counts)
MLP, following Bengio et al. 2003
CNN, following DeepMind WaveNet 2016 (in progress...)
RNN, following Mikolov et al. 2010
LSTM, following Graves et al. 2014
GRU, following Kyunghyun Cho et al. 2014
Transformer, following Vaswani et al. 2017

"""

words = open('names.txt', 'r').read().splitlines()
print("len(words)", len(words))
print("10 names, words[:10]: ", words[:10])

print("shortest word length: ",min(len(w) for w in words))
print("shortest word length: ",max(len(w) for w in words))

# what is all packed into word "isabella" ??

#=========================>
print(" <======================================== Line 28, starting Bigram Model ========================================>")
# bigram langugae model: 2 characters at a time. 
# we are looking at one character at a time, and trying to predict the next word.. 

for w in words[:2]:    
    
    print("current word", w)
    
    for ch1, ch2 in zip(w, w[1:]):
        
        print("functioning of zip, ends at shorter word, ch1, ch2: ", ch1, ch2)
        print("============")
        

bigram_hashmap = {}
for w in words[:3]:  
    
    # creating start token and end token
    chs = ['<S>'] + list(w) + ['<E>']
    
    print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        print("Printing Bigrams",ch1, ch2)
        
        # how often one word follow the other? ==> using count..
        bigram = (ch1, ch2)
        bigram_hashmap[bigram] = bigram_hashmap.get(bigram, 0)+1
        
print("bigram_hashmap", bigram_hashmap)

print(" <=============== Doing bigram hashmap for all the words =============================>")

bigram_hashmap = {}
for w in words:  
    
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['<S>'] + list(w) + ['<E>']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        #print("Printing Bigrams",ch1, ch2)
        
        # how often one word follow the other? ==> using count..
        bigram = (ch1, ch2)
        bigram_hashmap[bigram] = bigram_hashmap.get(bigram, 0)+1

#print("bigram_hashmap", bigram_hashmap)

sorted_by_count = sorted(bigram_hashmap.items(), key = lambda val:-val[1])
print(" ")
#print("sorted_by_count in decreasing order", sorted_by_count)

print("=========> storing bigram information in 2-D array, rows are going to be 1st character, \
      columns are going to be the second character., each entry will tell us how often second follows first")
      
a = torch.zeros((3,5), dtype = torch.int32)
print("a", a)
print("a.dtype", a.dtype)

#print(" ")
print("\n =====> creating bigger array now")

N = torch.zeros((28,28), dtype = torch.int32)

# need a lookup from characters to integers. 
print(len(set(''.join(words))))   # joing all the words and putting in a set. 

# sorted list
chars = sorted(list(set(''.join(words))))

# not having two tokens just have 1 token and have position 0 ('.')
# and offset all other letters by 1. 
# stoi = {s:i for i,s in enumerate(chars)}
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

for w in words:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['.'] + list(w) + ['.']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        N[ix1, ix2] +=1
        
        
print("N", np.array(N))
# Create a heatmap
# =============================================================================
# fig = go.Figure(go.Heatmap(z=N, colorscale='Viridis'))
# 
# # Customize the layout
# fig.update_layout(
#     title='Matrix N Heatmap',
#     xaxis_title='X Axis Label',
#     yaxis_title='Y Axis Label',
#     width=2000,  # Adjust the width of the plot
#     height=2000  # Adjust the height of the plot
# )
# =============================================================================

# =============================================================================
# # Add annotations (text) to the heatmap
# annotations = []
# for i, row in enumerate(N):
#     for j, value in enumerate(row):
#         annotations.append(
#             go.layout.Annotation(
#                 text=str(value),
#                 x=j,
#                 y=i,
#                 xref='x',
#                 yref='y',
#                 showarrow=False,
#                 font=dict(color='white', size=8)
#             )
#         )
# =============================================================================

# =============================================================================
# fig.update_layout(annotations=annotations)
# =============================================================================

# Render the plot in an external browser
# =============================================================================
# pio.renderers.default = "browser"
# fig.show()
# =============================================================================

#===========>
# creating better plots
itos = {i:s for s,i in stoi.items()}

plt.figure(figsize = (15,15))
plt.imshow(N, cmap = 'Blues')

# N[i,j].item() ==> gives the count N[i,j] is just the tensor
for i in range(27):
    for j in range(27):
        
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha = "center", va = "bottom", color = "gray")
        plt.text(j,i, N[i,j].item(), ha = "center", va = "top", color = "gray")
        
plt.axis("off")

print("============= We have all the information needed to sample from bigram character lagnguage model ==============")

# start sampling form the model. 
# grabbing the first row and 0th column..  this shows how many time each word is coming
# after the dot. 

print("first row and all columns",N[0,:])


# how to sample from this? ==> convert them to proabbilities. 
# by converting them to float and normalizing 
p = N[0].float()

p = p/p.sum()
print("Line 205, p",p)

# =============> Example #===============================>
#===================================================================>
# to sample from this distribution, we call torch.multinomial.
# give me proabbility distribution, i will sample and give integers. 
# example:
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator = g)
print("example p", p)  # sampled numbers between 0 and 1 (3 numbers) # example p tensor([0.7081, 0.3542, 0.1054])
# probability
p = p/p.sum()
print("probability p", p) # probability p tensor([0.6064, 0.3033, 0.0903])

# probability of the first element occuring is 60 5 of times.. from above example..
# tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]); 
# 0 occures 60 percent of times; 0, 1 and 2 only because of 3 in p = torch.rand(3)
print(torch.multinomial(p, num_samples=20, replacement = True, generator = g))

#===========================>
#===================================================================>
p = N[0].float()

# =============================================================================
# p tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
#         0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
#         0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290,
#         0.0000])
# =============================================================================
p = p/p.sum()

g = torch.Generator().manual_seed(2147483647)

ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()

print("index", ix)

print("Line 242: character drawn: ",itos[ix])

#==========================================================================>
#working on it now.. 


g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    ix = 0
    out = []
    while True:
        p = N[ix].float()
        p = p/p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()
        #print("Line 254: itos[ix]",itos[ix])
        out.append(itos[ix])
        if ix ==0:
            break
    # 1 answer: lamoynayrkiedengin. ==> Bigram language model is terrible
    print("Line 262: ",''.join(out))



#==============>
# another way fo doing this: (above thing is terrible)
p.shape

g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    ix = 0
    out = []
    while True:
        #p = N[ix].float()
        #p = p/p.sum()
        p = torch.ones(27)/27 # uniform distribution.. now samples from that. 
        
        ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()
        #print("Line 254: itos[ix]",itos[ix])
        out.append(itos[ix])
        if ix ==0:
            break
    
    # 1 answer: Line 286:  woflfjxflylgbegpjdpovdtw.
    # everything is equally likely.. (complete garbage.. so above one is terrible..)
    print("Line 286: ",''.join(out))

#==========================================================================>
# So working on the bigram model now.. (used it before also, )

# getting the matrix.. if the probabiltiies. 

g = torch.Generator().manual_seed(2147483647)

# adding smoothing factor.. we can add any factor # k is smoothign factor
k = 1
P = (N+k).float()

# we want to divide all the rows by their respective sums.. 
#P = P/P.sum(1)  ==> wrong
P = P/P.sum(1, keepdim = True) #==> (27, 1) vector


print("Line 303, Wrong, because this normalizes each row vector: P.sum(1).shape \
      ",P.sum(1).shape) #==========> Wrong 
      
print("Line 304, Correct, we want column vector to be normalized \
      : P.sum(1, keepdim = True).shape", P.sum(1, keepdim = True).shape) # ==> correct

print("Line 301, normalized P[0].sum()", P[0].sum())
# ===================>
# seeing broadcasting rules
# 27, 27 # P size
# 27, 1  # P.sum(1, keepdim = True) size

# one of the trailing dimension is 1. dimenion 0 is equal
# so the operation is braodcasting.. 
#=====================>

for i in range(20):
    ix = 0
    out = []
    while True:
        
        p = P[ix]
        #p = N[ix].float()
        #p = p/p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()
        #print("Line 254: itos[ix]",itos[ix])
        out.append(itos[ix])
        if ix ==0:
            break
    # 1 answer: lamoynayrkiedengin. ==> Bigram language model is terrible
    #print("Line 262: ",''.join(out))
    
    
#==========================================================================>
# Making it more efficient by inplace operatiosn Line 348

# getting the matrix.. if the probabiltiies. 

g = torch.Generator().manual_seed(2147483647)

# adding smoothing factor.. we can add any factor # k is smoothign factor
k = 1
P = (N+k).float()

# we want to divide all the rows by their respective sums.. 
#P = P/P.sum(1)  ==> wrong
P /= P.sum(1, keepdim = True) #==> (27, 1) vector


print("Line 303, Wrong, because this normalizes each row vector: P.sum(1).shape \
      ",P.sum(1).shape) #==========> Wrong 
      
print("Line 304, Correct, we want column vector to be normalized \
      : P.sum(1, keepdim = True).shape", P.sum(1, keepdim = True).shape) # ==> correct

print("Line 301, normalized P[0].sum()", P[0].sum())
# ===================>
# seeing broadcasting rules
# 27, 27 # P size
# 27, 1  # P.sum(1, keepdim = True) size

# one of the trailing dimension is 1. dimenion 0 is equal
# so the operation is braodcasting.. 
#=====================>

for i in range(20):
    ix = 0
    out = []
    while True:
        
        p = P[ix]

        ix = torch.multinomial(p, num_samples=1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix ==0:
            break
        
    print("Line 379: ",''.join(out))

print(" Line 381, we did bigram model by just counting.. Now how to evaluate \
       quality fo the model? Offcourse, Loss funciton. or Training Loss")
       
print("Line 384=============================================================")
# probability that model assigns to every one of the bigrams.
"""
Suppose you have a dataset with n independent samples {x1, x2, ..., xn},
 and a model with a set of parameters θ. The likelihood L(θ) is the 
 probability of observing this dataset given the model parameters θ. 
 Mathematically, we can write the likelihood as the joint probability 
 of observing all the samples in the dataset:

L(θ) = P(x1, x2, ..., xn | θ)

Since the samples are independent, we can express the joint probability 
as the product of the individual probabilities:

L(θ) = P(x1 | θ) * P(x2 | θ) * ... * P(xn | θ)

So, the likelihood is the product of the individual probabilities of 
the samples in the dataset. Maximizing the likelihood means finding the 
model parameters θ that make the observed data most probable.
"""

# =============================================================================
# for a good training model, product of the probability should be high. that is likelihood
# should be high
# .e:  0.0478
# em:  0.0377
# mm:  0.0253
# ma:  0.3899
# a.:  0.1960
# .o:  0.0123
# ol:  0.0780
# li:  0.1777
# iv:  0.0152
# vi:  0.3541
# ia:  0.1381
# a.:  0.1960
# .a:  0.1377
# av:  0.0246
# va:  0.2495
# a.:  0.1960
# =============================================================================
print("==========================line 425: Loss function==========================")

# =============================================================================
# 5. Goal, maximize likelihood of the data w.r.t model parameters (statistical modelling)
# 	--> equivalent to maximizing the log likelihood
# 	--> equivalent to minimizing the negative log likelihood
# 	--> equivalent to minimizing the average negative log likelihood
# =============================================================================


# for a good gtgraining model, likelihood should be high
# since we are fitting on it.. 
# but since likelihood is a multiplication, we can
# work with log likelihood

# Taken average negative log-likelihood as a  loss function

log_likelihood = 0.0
count = 0 # for normalizing/averaginig log likelihood, just for convenience
for w in words[:3]:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['.'] + list(w) + ['.']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        
        # probability that bigram assigns to each one of these..
        prob = P[ix1, ix2]
        
        # log (1) = 0 
        # as the number goes smaller, log probability will become more and more negative
        logprob = torch.log(prob)
        log_likelihood += logprob
        count+=1
        
        print(f'{ch1}{ch2}: {prob: .4f} {logprob: 4f}')

print(f'{log_likelihood = }')  
# generally in terms of loss low value means loss is low.. 
# but log likelihood is opposite.. 
# so we work with negative log likelihood

nll =-log_likelihood
print("Line 473",f'{nll = }')  # now it is 38, which is bad.. 
print("Line 474",f'{nll/count}') # average negative log-likelihood.. loss function

#=================>
print(" <-============ Line 477: Testing logic on andrejq =====================>")
# testing above logic on a word andrejq ==> jq proabbility is 0, so 
# negative log likelihood ==> is inf.. ==> which means very very bad.. 

log_likelihood = 0.0
count = 0 # for normalizing/averaginig log likelihood, just for convenience
for w in ["andrejq"]:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['.'] + list(w) + ['.']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        #print("Line 490: ch1, ch2", ch1,ch2)
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        
        # probability that bigram assigns to each one of these..
        prob = P[ix1, ix2]
        
        # log (1) = 0 
        # as the number goes smaller, log probability will become more and more negative
        logprob = torch.log(prob)
        log_likelihood += logprob
        count+=1
        
        print(f'{ch1}{ch2}: {prob: .4f} {logprob: 4f}')

print(f'{log_likelihood = }')  
# generally in terms of loss low value means loss is low.. 
# but log likelihood is opposite.. 
# so we work with negative log likelihood

nll =-log_likelihood
print("Line 512",f'{nll = }')  # now it is 38, which is bad.. 
print("Line 513",f'{nll/count}') # average negative log-likelihood.. loss function

#================>
# Infinitioy is not liked,. anywhere.. so we need to do model smoothign..
# we did smoothing:
    
# =============================================================================
#     # adding smoothing factor.. we can add any factor # k is smoothign factor
#     k = 1
#     P = (N+k).float()
# =============================================================================

#===============================================================================>
print(" print 531 ======================= Lets get to training based method ==========================")

# create training set of all the bigrams

# iterate over all bigrams

# training set is maade up of two lists.. 

# create the training set of bigrams (x,y); we are given the first character (x), 
# let's predict the next one (y)
xs, ys = [], []
# words[:1] ==> ['emma']
for w in words[:1]:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['.'] + list(w) + ['.']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        print("Line 554",f'{ch1}, {ch2}: {ch1} {ch2}')
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
print("xs", xs)
ys = torch.tensor(ys)
print("Line 561 ys", ys)

#=================================================
# xs tensor([ 0,  5, 13, 13,  1])
# ys tensor([ 5, 13, 13,  1,  0])
# 
# for xs ==> 5; want high probability for ys ==> 13

# for a particular value in tensor xs, we want thr proabability of 
# ys in the same row to be high
#===========================================================================

# common way of encoding integers is one hot encoding
# we cannot directly give the tensors as inputs. 

# in one-hot encoding, we create a vector where.. 
# all dimension of a vector is 0 except 13th dimension. (13th dimension is 1)

x_one_hot_encoded =  F.one_hot(xs, num_classes = 27)
print("Line 583, x_one_hot_encoded ", x_one_hot_encoded ) #(shape 5,27) beccause word is .emma

# 5 rows, each for 1 character
# 27 columns.. 
# each row will be a separate input into a neural network
plt.imshow(x_one_hot_encoded) # yellow means turned on. 

print("x_one_hot_encoded.dtype", x_one_hot_encoded.dtype)

# we do not want integers as an input to neural networks.. 
# input to onehot was integer mostly.. 
# we would have to cast it to float
# cannot pass integers as neural network input
x_one_hot_encoded =  F.one_hot(xs, num_classes = 27).float() #(5,27)
print("Line 597, x_one_hot_encoded.dtype", x_one_hot_encoded.dtype)

#====================================>
print(" ====================== Line 600: constructiing the first neuron on example word ",".emma","======================")

# iintiialize weights for the neuron from normal distribution
W = torch.randn((27,1))  # column vector of 27 numbers.. 

# weights have ot be multiplied by inputs

# (5,27) multiplie by (27,1) ==> (5,1) ==> this is because, neurons got activates
# at 5 places.. 
# (27,1) ==> 1 rerpesents 1 single neuron
# (5,27) # 5 represents the word of length 5. (bigrams) which are 5 ffor emma
# 27 ==> characters are 27. 
x_one_hot_encoded @ W  


#====================================>
print(" ====================== Line 616: constructiing the first neuron ======================")

# =============================================================================
# In deep learning and machine learning, a weight vector typically has two
#  dimensions corresponding to the input and the neurons in the next layer. 
#  
#  If the weight matrix were initialized to (5, 27), it would mean that 
#  each bigram in the input sequence would be connected to each character
#  in the output sequence. This may not be sufficient to capture the complex
#  relationships between the input and output, which is why a higher-dimensional
#  weight matrix is often used.
# =============================================================================


# iintiialize weights for the neuron from normal distribution
# first dimension initilised to 27 and not 5 ==> becasuse
# even though current input is 5, we can have 27 inputs (input bigrams in total)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g)  # second simension 27 ==> represents 27 neurons 

# weights have ot be multiplied by inputs

# (5,27) multiplie by (27,27) ==> (5,27) ==> this is because, 27 neurons got activated
# at 5 places.. 
# (27,27) ==> 27 rerpesents 27 single neuron (2nd dimension)
# (5,27) # 5 represents the word of length 5. (bigrams) which are 5 ffor emma; 27 ==> characters are 27. 
#

# all 27 neurons will be evaluated on all 5 inputs
x_one_hot_encoded @ W  # shape ==> (5,27)
#print("Line 645: x_one_hot_encoded @ W", x_one_hot_encoded @ W)
# x_one_hot_encoded @ W ==> tells us that for every one of 27 neurons, 
# what is the firing rate of those neurons 
# on every one of those 5 exampesl

# x_one_hot_encoded @ W [3,13] ==> 0.5008, gives the firing rate of 13th neuron for 3rd input bigram
print("Line 651, x_one_hot_encoded @ W [3,13] ", (x_one_hot_encoded @ W) [3,13] )
# same result as line 651
print("Line 653, (x_one_hot_encoded[3]*W[:,13])[13]",(x_one_hot_encoded[3]*W[:,13])[13])


# second dimension is kept to be 27, because we want some kind of probability distirbution
# over 27 numbers.. (this pporbability distribution through neurons)

# 27 numbers from neural net will give us log counts
# to get the counts, we will exponentiate these log counts ( exp(log(counts)))

# so x_one_hot_encoded @ W ==> gives log counts.. element wise exponentiate them to get counts
# (x_one_hot_encoded @ W).exp() ==> represent counts
# negative ==> below 1
# positive ==> above 1
# =============================================================================
# one row looks like:
# [ 0.8328,  1.3358,  1.0231,  2.5768,  0.1399,  2.3167,  0.8860,  0.2850,
#   1.3993,  0.1648,  0.3587,  1.3945,  0.3855,  0.4647,  1.0331,  2.4491,
#   1.2715,  1.3447,  0.4658,  3.9770,  0.6586,  0.3566,  1.7517,  1.5386,
#   6.2145,  1.5763,  0.1977]
# =============================================================================
    
print("Line 674, counts, (x_one_hot_encoded @ W).exp()", (x_one_hot_encoded @ W).exp())

# logits ==> neural network output which is log count
logits = x_one_hot_encoded @ W    # log counts #(5,27) 


counts = logits.exp()    # counts #(5,27)  

# counts.sum(1, keepdims = True) ==> (5,1) 
# 5 numbers one for each row, extend to all 27 numbers in a row

# counts.sum(1, keepdims = True) ==> normalizing rows to form the probability distribution
probs = counts/counts.sum(1, keepdims = True) # counts normalized

print("probs[0].sum()",probs[0].sum() ) #===> should be 1. (all probabilities sum up to 1)

#==============================>
print(" ======================== Line 692, rervising Neural network part ===================")

xs
ys

g = torch.Generator().manual_seed(2147483647+30)

# requires_grad = True ==> means we want to calculate the gradients
W = torch.randn((27,27), generator = g, requires_grad = True)  # second simension 27 ==> represents 27 neurons 

#============>
# Forward pass
x_one_hot_encoded =  F.one_hot(xs, num_classes = 27).float() #(5,27)
print("Line 701, x_one_hot_encoded.dtype", x_one_hot_encoded.dtype)

logits = x_one_hot_encoded @ W    # log counts #(5,27) 

# these last two lines are basically the softmax 

counts = logits.exp()    # counts #(5,27)  
probs = counts/counts.sum(1, keepdims = True) # counts normalized

#===========>
# loss function part
# for 5 bigrams making up ".emma" index of e==> 5, index of m ==> 13
nlls = torch.zeros(5)
print("----------------------------- Just to showcase loss function, just an example, not used: Line 715 -------------------------------")
for i in range(5):
    
    x = xs[i].item()   # input character index
    y = ys[i].item()   # label character index
    
    print("-------------------------------- Line 721 ---------------------------------------")
    print(f'bigram example {i+1}: {itos[x]}{itos[y]}, (indexes {x}, {y})')
    print('input too the nerural net: ', x)
    print("output probabilities from the neural net: ", probs[i])
    print("label (actual nnext character):", y)
    p = probs[i,y]
    
    print("probability assignes by the net to the correct character: ", p.item())
    
    logp = torch.log(p)
    
    print("log likelihood: ", logp.item())
    
    nll = -logp
    
    print("negative log likelihood: ", nll.item())
    
    nlls[i] = nll
    
    
print("===============")
print("average negative log likelihood, i.e loss =", nlls.mean().item())

#======================================================================================>
print("================= line 746, gradient part, loss is calcualted again =========")
# to minimize loss, we have to tune W by computing gradients.. 

# Calcualte loss. 

# xs tensor([ 0,  5, 13, 13,  1])
# ys tensor([ 5, 13, 13,  1,  0])

# =============================================================================
# we are interested in values of ys
# in first example, we are interested in the value at location 5 (e) when first char was "."
# in second example, we are interested in the value at location 13(m) when fiirst char was "e"
# and so on. 
#  
# so interested in followisn:
# =============================================================================
    
print("Line 762: We want the following from ys: probs[0,5], probs[1,13], probs[2,13], probs[3,1], probs[4,0]",
      probs[0,5], probs[1,13], probs[2,13], probs[3,1], probs[4,0])


# =============================================================================
# # probability that bigram assigns to each one of these..
#         prob = P[ix1, ix2]
#         
#         # log (1) = 0 
#         # as the number goes smaller, log probability will become more and more negative
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         count+=1
#         
#         print(f'{ch1}{ch2}: {prob: .4f} {logprob: 4f}')
# 
# print(f'{log_likelihood = }')  
# # generally in terms of loss low value means loss is low.. 
# # but log likelihood is opposite.. 
# # so we work with negative log likelihood
# 
# nll =-log_likelihood
# print("Line 512",f'{nll = }')  # now it is 38, which is bad.. 
# print("Line 513",f'{nll/count}') # average negative log-likelihood.. loss function
# =============================================================================

#torch.arange(5)
# probs dimension: #(5,27) 
loss = -probs[torch.arange(5), ys].log().mean()

# say loss value here is 3.76
print("line 792, computed loss", loss)  # 3.3080,
print("================= line 746, Manual Backward pass and then Manual forward pass ==============")

# Backward pass
# initially set gradients to None
W.grad = None  # lack of gradient.. #(27,27)
loss.backward()
print("Line 802: after backward prop, W.grad.shape", W.grad.shape)

# update to weight
W.data += -0.1*W.grad

# =============================Doind forward pass again..============================== 
# loss should be lesser now.. 
#  loss  decreased from line 793 after doing forward pass again..  value here is 3.76
x_one_hot_encoded =  F.one_hot(xs, num_classes = 27).float() #(5,27)
print("Line 701, x_one_hot_encoded.dtype", x_one_hot_encoded.dtype)

logits = x_one_hot_encoded @ W    # log counts #(5,27) 

# these last two lines are basically the softmax 

counts = logits.exp()    # counts #(5,27)  
probs = counts/counts.sum(1, keepdims = True) # counts normalized
loss = -probs[torch.arange(5), ys].log().mean()
print("line 820, computed loss reclaculated", loss) # tensor(3.2884


#===============================================================================>
print(" print 825 ======================= Rearraging things again, calculating for all the words ==========================")

# create training set of all the bigrams

# iterate over all bigrams

# training set is maade up of two lists.. 

# create the training set of bigrams (x,y); we are given the first character (x), 
# let's predict the next one (y)
xs, ys = [], []
# words[:1] ==> ['emma']
# now using all the words
for w in words:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['.'] + list(w) + ['.']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        #print("Line 848",f'{ch1}, {ch2}: {ch1} {ch2}')
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
#print("xs", xs)
ys = torch.tensor(ys)

num = xs.nelement()
print("number of examples: ", num)

# intialize the network
g = torch.Generator().manual_seed(2147483647+30)

# requires_grad = True ==> means we want to calculate the gradients
W = torch.randn((27,27), generator = g, requires_grad = True)  # second simension 27 ==> represents 27 neurons 

# regularizing W
(W**2).mean()
# Gradient Descent
print("Line 866, 10 iterations of gradient descent")
for k in range(10):
    x_one_hot_encoded =  F.one_hot(xs, num_classes = 27).float() #(5,27)
    
    logits = x_one_hot_encoded @ W    # log counts #(5,27) 
    counts = logits.exp()    # counts #(5,27)  
    probs = counts/counts.sum(1, keepdims = True) # counts normalized
    loss = -probs[torch.arange(num), ys].log().mean() +0.01*(W**2).mean()
    
    print("Line 875, loss.item()",loss.item())
    # Backward pass
    # initially set gradients to None
    W.grad = None  # lack of gradient.. #(27,27)
    loss.backward()
    #print("Line 802: after backward prop, W.grad.shape", W.grad.shape)

    # update to weight
    W.data += -50*W.grad

#==========================================================>
print(" ============================ print 899 : Sampling from Neural Net model ========================")
g = torch.Generator().manual_seed(2147483647+30)

for i in range(5):
    
    out = []
    ix = 0
    while True:
        
        # BEFORE
        p = P[ix]
        
        # Now p comes from neural net
        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()
        p = counts/counts.sum(1, keepdims = True)  # probabilities for next character
        
        
        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[ix])
        
        if ix == 0:
            break
        
        print(''.join(out))
        
        