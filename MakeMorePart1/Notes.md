Terms:

1. Braodcasting

2. Likelihood : product of the proabbilities

3. simple bigram model training. simmply generate probabilities of the count of bigrams

4. log probabilities are negative. 

5. Goal, maximize likelihood of the data w.r.t model parameters (statistical modelling)
	--> equivalent to maximizing the log likelihood
	--> equivalent to minimizing the negative log likelihood
	--> equivalent to minimizing the average negative log likelihood
	
6. Model smoothing with fakecounts; infitinity is not liked

7. cannot pass integers as neural network input
8. logits (or W) are the log_counts not counts
9. to get counts, we have to exponentiate log counts (i.e logits)

10. Softmax: output of neural network is like log counts, 
   exponentiate it to get into form of counts, probability is the count normalized
   
11. Optimizing a Neural network. 
12. instead of taking a character to predict next, if we take many characters to predict, next
	fundamentally only logits calcaculation will change.  
	i.e from code: logits = x_one_hot_encoded @ W    # log counts #(5,27) # nothing else
	
13. if W's are all equal to each other in say 
	W = torch.randn((27,27)) say W are initialized to 0. then logits are 0 and counts = exp(0) = 1
	and probabilities will be completely uniform . this is a kind of regularization
	
14. but instead of doing 13, we add a regularizer to loss (regularizer ==> 0.01*(W**2).mean())
    # loss = -probs[torch.arange(num), ys].log().mean() +0.01*(W**2).mean()

	
