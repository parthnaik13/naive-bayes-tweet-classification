# Naive Bayes approach to classify tweets to north american cities

Naive Bayes Model of predicting the city given the tweet:
1. We parse the training data to get the tweets and corresponding labels.
2. We pre-process the data (more mentioned below).
3. Then we convert the tweets into a bag of words model which is in the form of a dictionary.
   The dictionary has key values as the cities and the values for these keys are dictionaries having keys as the words and values as the number of 
   times the words occur in the city tweets.
4. Using this bag of words we calculate the priors and likelihoods
   priors: P(City) = # times city occurs in training tweets / # of tweets
   likelihoods : P(word|city) = # word occurs in the city's tweets / # of tweets for that city
5. We then calculate the accuracy on training data by classifying the data.
   We calculate P(City|W1,..,Wn) =(approx) P(W1,..,Wn|City) * P(City)		
   PS : We ignore the denominator as to find the max we can ignore the denominator as the denominator is postive as it is a probability.
   Now we use the naive bayes assumption:
   P(W1,..,Wn|City) = P(W1|City)*P(W2|City)*...*P(Wn|City)
6. Finally we parse testing data and calculate the testing error.
7. We then write our output to the file.

Tweet pre-processing :
1. We remove the stop words from the tweets.
2. Additionally we also remove some undesired words like '#jobs' and 'I'm' which were not covered in the stop words.
3. Removing symbols increased the accuracy.
4. Making the words lower case actually decreased the accuracy so not doing it.

Laplace smoothing:
While calculating likelihood for a word i.e P(Word|City) for a word that has not yet occured in the city's tweets the P(Word|City) = 0 seems logical,
but we need to assign some value however small to the probability such that it is not 0 as there is always a very remote possibilty that the word might
occur in the city's tweets.

Also if we assign 0 the probabilty the whole product will be 0 which is not at all desirable.

<b>The test accuracy of the classifier comes out to be 70%. Its quite hard to classify tweets based on just the occurence of words!</b>

How to run:
1. Clone the repository
2. Execute the naive bayes classifier - python geolocate.py tweets.train.txt tweets.test1.txt output.txt
The output file has the format - predicted_city, true_city, tweet
