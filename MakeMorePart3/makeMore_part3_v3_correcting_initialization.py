

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
#%matplotlib inline



# read in all the words
words = open('names.txt', 'r').read().splitlines()
words[:8]

print(len(words))

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

#%% Train, Validation and Test dataset

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
          ix = stoi[ch]
          X.append(context)
          Y.append(ix)
          context = context[1:] + [ix] # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

#%% 

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g)
W2 = torch.randn((n_hidden, vocab_size),          generator=g)
b2 = torch.randn(vocab_size,                      generator=g) *0 

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
#%% 

# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
      # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
    
    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer
    hpreact = embcat @ W1 + b1 # hidden layer pre-activation
    h = torch.tanh(hpreact)  # hidden layer
    
    # how can we achieve logits at initialization to be more
    # closer to zero.. (or same)
    # b2 should be zero at initialization, we do not want to add bias
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function
    
    # backward pass
    # backward pass
    for p in parameters:
        p.grad = None
        
    loss.backward()
    
    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr*p.grad
      
    if i %10000 == 0: #print every once in a while

        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

    lossi.append(loss.log10().item())
    
    break
    
#%% loss progression 
#==> inital step loss is way too high (27). so something
# is wrong with setting up at initialization. 

# initillly probability of all the characters should be equal, since we are starting
# so it should be a uniform distribution. 

# so initially uniform probability ==> 1/27
# so loss should be -torch.tensor(1/27.0).log() ==> 3.2958 ==> that should be the initial value for loss. 

# very high loss because network is very confidently wrong. 
# =============================================================================
#       0/ 200000: 27.8817
#   10000/ 200000: 2.8240
#   20000/ 200000: 2.5163
#   30000/ 200000: 2.8836
#   40000/ 200000: 2.0655
#   50000/ 200000: 2.4969
#   60000/ 200000: 2.4992
#   70000/ 200000: 2.0261
#   80000/ 200000: 2.4461
#   90000/ 200000: 2.2756
#  100000/ 200000: 2.0263
#  110000/ 200000: 2.3349
#  120000/ 200000: 1.8987
#  130000/ 200000: 2.3830
#  140000/ 200000: 2.1785
#  150000/ 200000: 2.1847
#  160000/ 200000: 2.0797
#  170000/ 200000: 1.8553
#  180000/ 200000: 1.9629
#  190000/ 200000: 1.8522
# =============================================================================

#%% Manual, very manual: looking at initialization wrong with the example.. 

# say just have 4 charcaters.. 
# over 4 characters, any thing can come out, since it is exact same loss for all
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = torch.softmax(logits, dim = 0)
loss = -probs[2]. log()  # label ==> 2, it can be anyone of 3. i am calculating looss wr.t. to label 2


print("Line 154: probs, loss: ", probs, loss)

#case 1: when label value is high: loss is low

logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
probs = torch.softmax(logits, dim = 0)
loss = -probs[2]. log()  # label ==> 2, it can be anyone of 3. i am calculating looss wr.t. to label 2
print("Line 163: probs, loss: ", probs, loss) # tensor(0.0200)

#case 2: when label value is low, loss is high

logits = torch.tensor([0.0, 5.0, 0.0, 0.0])
probs = torch.softmax(logits, dim = 0)
loss = -probs[2]. log()  # label ==> 2, it can be anyone of 3. i am calculating looss wr.t. to label 2


print("Line 172: probs, loss: ", probs, loss) #tensor(5.0200)

print("Line 174: Point is that logits have to be equal when they are initialized, means loss will not be that high")

#%% Checking the loss after training. 
plt.plot(lossi)


@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    x,y = {
      'train': (Xtr, Ytr),
      'val': (Xdev, Ydev),
      'test': (Xte, Yte),
    }[split]
    emb = C[x] # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
    hpreact = embcat @ W1 + b1

    h = torch.tanh(hpreact) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

#%% Sampling from the model ==> much nicer looking woprds..

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # forward pass the neural net
      emb = C[torch.tensor([context])] # (1,block_size,n_embd)
      h = torch.tanh(emb.view(1,-1) @ W1 + b1)
      
      logits = h @  W2 + b2
      


      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
    
#%%


#============>
# there is a lot of noise in the steps (it is thick).. one reason might be that batch size is quite low for trianing.. 

# visualize the embeddings trained by the neural network.. 
plt.figure(figsize = (8,8))
plt.scatter(C[:,0].data, C[:,1].data, s = 200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha = "center", va = "center", color = "white")

plt.grid('minor')
