
import torch
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
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
stoi = {s:i for i,s in enumerate(chars)}

stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    # creating start token and end token list(w) converts string to a list of characters
    chs = ['<S>'] + list(w) + ['<E>']
    
    #print("current word: ", w, ", Token: ",chs)
    
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        N[ix1, ix2] +=1
        
        
print("N", np.array(N))
# Create a heatmap
fig = go.Figure(go.Heatmap(z=N, colorscale='Viridis'))

# Customize the layout
fig.update_layout(
    title='Matrix N Heatmap',
    xaxis_title='X Axis Label',
    yaxis_title='Y Axis Label',
    width=2000,  # Adjust the width of the plot
    height=2000  # Adjust the height of the plot
)

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
pio.renderers.default = "browser"
fig.show()

#===========>
# creating better plots
itos = {i:s for s,i in stoi.items()}

plt.figure(figsize = (15,15))
plt.imshow(N, cmap = 'Blues')

# N[i,j].item() ==> gives the count N[i,j] is just the tensor
for i in range(28):
    for j in range(28):
        
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha = "center", va = "bottom", color = "gray")
        plt.text(j,i, N[i,j].item(), ha = "center", va = "top", color = "gray")
        
plt.axis("off")




