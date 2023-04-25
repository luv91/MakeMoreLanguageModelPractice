
1. creation of embedding table for 27 characters: C = torch.randn((27,2))

2. Pytorch indexing is awesome
	a. For example, embeddingC[X], X is of shape [5,3] works --> C[X] shape is torch.Size([5, 3, 2])
	
3. boradcasting explained: Minute 29.00 (http)

### In step 4, all the steps from b to g can be obtained by just using F.cross_entropy(logits, Y)
### steps b to g are inefficeint, 1. because pytorch do not create all these inbetween tensors and
### backpropagation is much more efficient that way. 
### forward pass is also much more efficient 
4. process of getting the loss

	a. get the logits: logits = hidden @ W2 + b2
	
	b. exponentiate the logits to get the count: counts = logits.exp()
	   --> this step is inefficinet: torch.tensor([-100, -3, 0, 100]) will return tensor([1e-44, 9.1e-2, 1, inf])
	   --> the last element ran out of range and that's why infiinity. 
	
	c. convert count to probability: prob = counts/counts.sum(1, keepdims = True)
	d. getting the probabilities: probabilities = prob[torch.arange(emb.shape[0]), Y]
	e. getting the log probabilities: log_probabilities = prob[torch.arange(emb.shape[0]), Y].log()
	f. get the average log probabilities: average_log_probabilities = prob[torch.arange(emb.shape[0]), Y].log().mean()
	g. create negative log likelihood loss:  negative_log_likelihood_loss = -prob[torch.arange(emb.shape[0]), Y].log().mean()