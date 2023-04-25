### Problem 1: Initialization losses are very high

1. At initialization: logits should have the same value (uniform distribution), so loss shouldn't be very high.

2. output logits ==> logits = h @ W2 + b2 # output layer;

    a. b2 should be zero at initialization.
    b. W2 should be smaller, so multiplied by 0.1 ==> W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
        But why not zero initially?

Because of a and b, the initial loss was much lower.

### Problem 2: Many output of the first hidden layer output is 1 torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)

1. tanh squashes numbers between -1 and 1, but many values are -1 and 1 due to tanh.

2. This is a huge problem for during the backpropagation because we will propagate through torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)

    a. If many of them are near -1 or 1, their gradient will be killed. We will not backpropagate through
        - Those units where values are extreme (-1,1) in the forward pass.
        - Because self.grad += (1-t**2)*out.grad ==> t is (-1, 1) ==> self.grad = 0
        
        - These things are not optimal. How to fix this?
        - hpreact = embcat @ W1 + b1 ==> this is too far from 0, and thus causing the issue.
        
        ##### Answer is squashing initial W1 and bias b1:
        - W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) *0.1
        - b1 = torch.randn(n_hidden, generator=g) * 0.01
        
3. Variance increases or decreases, that is if the weights, let us say are following Gaussian
   so their distribution can shrink or expand a lot. 
   But we want their variance to be constant, do not change much.

    a. x = torch.randn(1000,10)
    b. w = torch.randn(10,200) # 10 inputs, 200 neurons in the hidden layer
    c. y = x @ w
    d. print(x.mean(), x.std())
    e. print(y.mean(), y.std())

    ### Initial values:

    g. ######## tensor(-0.0185) tensor(1.0030)
    h. ######## tensor(-0.0028) tensor(3.2714) ==>
    i. ######## y's standard deviation has expanded from 1.0030 to 3, means Gaussian is expanding, and we do not want that.

    ### How can we preserve this? Not let the thing expand?
    ### Answer: is to divide the weight by the square root of input; w = torch.randn(10,200) --> becomes --> w = torch.randn(10,200)/(10**0.5)	
	
	j. most common wy of initialization is kaiming_normal
	

		
