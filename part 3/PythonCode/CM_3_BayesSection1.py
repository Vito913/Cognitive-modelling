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
def bayesFunctionMultipleHypotheses(prior, likelihood):
    for i in range (len(prior)):
        H = prior[i]
        #notH = 1 - H
        huidig = likelihood[i]
        DH = huidig[0]
        DnotH = huidig[1] # <- This implies you give DnotH as input which I dont think you necessarily do so this is wrong still!!
        #DnotH = DandnotH/ notH

        # P (!A, B) = P (B)P (!A|B)
                   #= P (!A)P (B|!A)

        print("bayesFunctionMultipleHypotheses: ", bayesFunction(H, DH, DnotH))

bayesFunctionMultipleHypotheses([0.1], [[0.9, 0.3]])

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
def bayesFactor(posteriors, priors):
    #DH = posteriors[0]
    # First compare DH to DnotH, which idk how to get. << MISSING!
    for i in range (len(priors)):
        H = priors[i]
        notH = 1 - H
        #print("not H: ", notH) 
        priorodds = H/ notH
        #print(i, " and prior:", priorodds)
        DH = posteriors[i]
        DnotH = 1 - DH
        posteriorodds = DH / DnotH
        #print(i, " and posterior:", posteriorodds)
        #print(i, " and prior:", priorodds)
        bayesF = posteriorodds / priorodds
        print("BayesFactor: ", bayesF)
        # Hij werkt voor de eerste input, maar niet voor de andere twee... WHY??!

# posterior odds = prior odds * bayesfactor
# bayes factor = posterior odds / prior odds

bayesFactor([0.85,0.05,0.1],[0.2,0.6,0.2]) # Where posteriors is always a list of lists...


# Implement, as	function called	“bayesFactor”. The function	takes as input 2 vectors:
# • A vector of	posteriors (e.g., (P(H1 | D), P(H2 | D), P(H3 |	D))	if there are 3 – but in	principle it could be N posteriors).
# • A vector of	priors (e.g., (P(H1), P(H2), P(H3))	if there are 3)
# It gives as output different Bayes Factors (for different comparisons), see below	for examples.
# Important: Make sure that:
# • The	order in which you give posteriors matches the order in which you give priors
# • The	first item in each vector relates to the item of interest P(“a person is an	AI student” | Data) and	(e.g., P(“a	person is an AI	student”)
# • When you give input	for	the	priors,	that the sum of	all	priors equals to 1
# • The	posteriors sum to 1


