
### Problem 1: Initialization losses are very high

1. At initialization: logits should have same value (uniform distribution), so loss shouldn't be very high

2. output logits ==> logits = h @ W2 + b2 # output layer; 
	
	a. b2 should be zero at initialization. 
	b. W2 should be smaller, so multiplied by 0.1 ==> W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
		byt why not zero initilly?
		
		
	because of a and b, initial loss was much lower .. 
	
### Priblem 2: many output of first hidden layer output is 1 torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)

1. tanh squashes number between -1 and 1. but many values are  -1 and 1. due to tanh.

2. this is a huge problem for during the backproagation. because we will propagate through torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)

	a. if many of them are near -1 or 1, their gradient will be killed. we will not backpropagate through 
		--> those units where values are etreme (-1,1) in forward pass. 
		--> because self.grad += (1-t**2)*out.grad ==> t is (-1, 1) ==> self.grad = 0
		
		--> these things are not optimal. How to fix this?
		--> hpreact = embcat @ W1 + b1  ==> this is too far from 0, and thus causing the issue.
		
		##### Answer is squasshin intial W1 and bias b1:
		--> W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) *0.1
		--> b1 = torch.randn(n_hidden,                        generator=g) * 0.01
		
		
		
3. variance increases or decreases, that is if the weights, let us say are following gaussian
   so their distribution can shrink or expand a lot. 
   but we want their variance to be constant, do not change much. 
   
	
	a. x = torch.randn(1000,10)
	b. w = torch.randn(10,200) # 10 inputs, 200 neurons in the hidden layer

	c. y = x @ w
	print(x.mean(), x.std())
	print(y.mean(), y.std())

	###### intial values:
	######## =============================================================================
	######## tensor(-0.0185) tensor(1.0030)
	######## tensor(-0.0028) tensor(3.2714) ==>
    ########	y's standard deviation has explanded from 1.0030 to 3, means gaussian is explanding. and we do not want that. 
	
	### how can we preserve this? not let the thing expand?
	##### answer: is to divide weight by square root of input; w = torch.randn(10,200) becomes w = torch.randn(10,200)/(10**0.5)

		
