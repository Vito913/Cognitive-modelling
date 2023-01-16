#
# Engineering 1.1:
#
def bayesFunction(H, DH, DnotH):
    DandH = H * DH
    notH = 1 - H
    DandnotH = notH * DnotH
    D =    DandH    + DandnotH
    HD = (H * DH)/ D
    return HD

print("BayesFunction: ", bayesFunction(0.1, 0.9, 0.3))

#
# Engineering 1.2:
#
#def bayesFunctionMultipleHypotheses(prior, likelihood):
 #   for i in range (len(prior)):
   #     H = prior[i]
        #notH = 1 - H
   #     huidig = likelihood[i]
   #     DH = huidig[0]
   #     DnotH = huidig[1] # <- This implies you give DnotH as input which I dont think you necessarily do so this is wrong still!!
        #DnotH = DandnotH/ notH

        # P (!A, B) = P (B)P (!A|B)
                   #= P (!A)P (B|!A)

    #    print("bayesFunctionMultipleHypotheses: ", bayesFunction(H, DH, DnotH))

def bayesFunctionMultipleHypotheses(priors, conditionals):
    # Make sure the length of the priors vector matches the length of the conditionals vector
    if len(priors) != len(conditionals):
        raise ValueError("The length of the priors vector must match the length of the conditionals vector.")
    # Initialize an empty list to store the posterior probabilities
    posteriors = []
    # Iterate through the hypotheses
    for i in range(len(priors)):
        # Compute the posterior probability using Bayes' rule
        posterior = (priors[i] * conditionals[i]) / sum([priors[j] * conditionals[j] for j in range(len(priors))])
        # Append the posterior probability to the list
        posteriors.append(posterior)
    return posteriors

print("bayesFunctionMultipleHypotheses: ", bayesFunctionMultipleHypotheses([0.4, 0.3, 0.3], [0.99, 0.9, 0.2])) # <- this one doesnt work.

# Implement	a second version of	Bayes’ rule	(equation 1.1) that	can	consider multiple hypotheses.	
# Call the function	“bayesFunctionMultipleHypotheses”
# To do	this, give two arguments as	input:
# A	vector of prior	probabilities of all possible hypotheses.
# A	vector of all likelihood functions of the data given these hypotheses.
# Important: Make sure that:
# The order	in which you give priors matches the order in which	you give conditionals
# The first	item in	each vector	relates	to the item	of interest (e.g., P(“a	person is an AI	student”) and P(Data | “a person is	an AI student”))

#
# Engineering 1.3:
#
# def bayesFactorOld(posteriors, priors):
#     #DH = posteriors[0]
#     # First compare DH to DnotH, which idk how to get. << MISSING!
#     for i in range (len(priors)):
#         H = priors[i]
#         notH = 1 - H
#         #print("not H: ", notH) 
#         priorodds = H/ notH
#         #print(i, " and prior:", priorodds)
#         DH = posteriors[i]
#         DnotH = 1 - DH
#         posteriorodds = DH / DnotH
#         #print(i, " and posterior:", posteriorodds)
#         #print(i, " and prior:", priorodds)
#         bayesF = posteriorodds / priorodds
#         print("BayesFactor: ", bayesF)
#         # Hij werkt voor de eerste input, maar niet voor de andere twee... WHY??!

# # # posterior odds = prior odds * bayesfactor
# # # bayes factor = posterior odds / prior odds

# print("bayesFactor OLD!!!: ", bayesFactorOld([0.9,0.05,0.05],[0.2,0.6,0.2]))


def bayesFactor(posteriors, priors):
    # Make sure the length of the posteriors vector matches the length of the priors vector
    if len(posteriors) != len(priors):
        raise ValueError("The length of the posteriors vector must match the length of the priors vector.")
    # Make sure the sum of all priors equals 1
    if sum(priors) != 1:
        raise ValueError("The sum of all priors must equal 1.")
    # Make sure the posteriors sum to 1
    if sum(posteriors) != 1:
        raise ValueError("The posteriors must sum to 1.")

    # Initialize an empty list to store the Bayes Factors
    bayes_factors = []
    # Iterate through the hypotheses
    H = priors[0]
    notH = 1 - H
    #print("not H: ", notH) 
    priorodds = H/ notH
    #print(i, " and prior:", priorodds)
    DH = posteriors[0]
    DnotH = 1 - DH
    posteriorodds = DH / DnotH
    #print(i, " and posterior:", posteriorodds)
    #print(i, " and prior:", priorodds)
    bayesF = posteriorodds / priorodds
    print("BayesFactor 1 vs not 1: ", bayesF)


    for i in range(len(posteriors)):
        # Initialize an empty list to store the Bayes Factors for current hypothesis i
        bayes_factor = []
        for j in range(len(posteriors)):
            if i!=j:
                # Compute the Bayes Factor
                bf = posteriors[i] / priors[i] / (posteriors[j] / priors[j])
                # Append the Bayes Factor to the list
                bayes_factor.append("BF " + str(i+1) + " vs " + str(j+1) + " : " + str(bf))
        bayes_factors.append(bayes_factor)
    return bayes_factors

print("bayesFactor: ", bayesFactor([0.85,0.05,0.1],[0.2,0.6,0.2]))
# Where posteriors is always a list of lists...

# Implement, as	function called	“bayesFactor”. The function	takes as input 2 vectors:
# • A vector of	posteriors (e.g., (P(H1 | D), P(H2 | D), P(H3 |	D))	if there are 3 – but in	principle it could be N posteriors).
# • A vector of	priors (e.g., (P(H1), P(H2), P(H3))	if there are 3)
# It gives as output different Bayes Factors (for different comparisons), see below	for examples.
# Important: Make sure that:
# • The	order in which you give posteriors matches the order in which you give priors
# • The	first item in each vector relates to the item of interest P(“a person is an	AI student” | Data) and	(e.g., P(“a	person is an AI	student”)
# • When you give input	for	the	priors,	that the sum of	all	priors equals to 1
# • The	posteriors sum to 1


