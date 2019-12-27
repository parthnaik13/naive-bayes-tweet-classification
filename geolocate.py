#!/usr/bin/env python3
# Naive Bayes tweet classification
# Author : Parth Naik

"""
Comments section : 
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

"""

import pdb
import sys
import string
import operator

# List of the 12 cities used to parse the input so that each tweet can be properly stored according to the labels. PS : Some tweets are multiline.
cities = ['Los_Angeles,_CA','San_Francisco,_CA','Manhattan,_NY','San_Diego,_CA','Houston,_TX','Chicago,_IL','Philadelphia,_PA','Toronto,_Ontario','Washington,_DC','Atlanta,_GA','Boston,_MA','Orlando,_FL']

"""Reference for stopwords - https://gist.github.com/sebleier/554280"""
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# Function to remove stop words and undesired words.
def removeStopWords(line):
	# List of undesired words obtained by looking at the top most common words for the cities.
	undesired = ['','I',"Im",'hiring','Hiring','New','/','The','Job','Jobs','job','__','CareerArc','St','bw','amp','latest','This']
	word_tokens = line.split(" ")
	#word_tokens = [w for w in word_tokens if not w in undesired]
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	
	for i in range(1,len(filtered_sentence)):
		# Reference - https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python 
		filtered_sentence[i] = filtered_sentence[i].translate(str.maketrans('','',string.punctuation))	# Remove punctuation
		#filtered_sentence[i] = filtered_sentence[i].lower()	Make words lowercase
		
	for word in filtered_sentence[1:]:
		if not word:
			filtered_sentence.remove(word)
			
	filtered_sentence = [w for w in filtered_sentence if not w in undesired]
	
	return filtered_sentence
	

# Parse training/testing file
def parseTweets(filename):
	with open(filename,'r',errors='ignore') as file:
		lines = file.readlines()

	# The labels along with the respective tweets
	labels = []
	tweets = []

	for line in lines:
		line = line.rstrip()
		words = removeStopWords(line)
		if not words:
			continue
		city = words[0]
		if city in cities:
			labels.extend([city])
			tweets.extend([words[1:]])
			# Store the current tweet line as the tweet can be multiline
			latest_tweet_line = words[1:]
		else:
			# If tweet dosen't start with a city then this line is the part of the previous ongoing tweet
			multi_line_tweet = latest_tweet_line + words
			# Remove old tweet
			tweets.pop()
			# Add new multiline tweet
			tweets.extend([multi_line_tweet])
			
	return labels,tweets

# Function to convert the tweets into a bag of words model 
def bagOfWordsModel(labels,tweets):
	# Our bag of words
	bag = {}

	# Initialize basic dictionary to store the words
	for city in cities:
		bag[city] = {}

	for i in range(0,len(labels)):
		city = labels[i]
		words = tweets[i]
		for word in words:
			if word in bag[city].keys():
				bag[city][word] += 1
			else:
				bag[city][word] = 1
	return bag

# Function to compute the priors of the cities and likelihoods of the words given cities.
def computeProbabilities(bag,labels):
	# Dict to store city and its count
	city_count = {}
	
	# Initialize city_count dictionary
	for city in cities:
		city_count[city] = 0
	
	# Get count of each city label in the given data
	for city in labels:
		if city in city_count.keys():
			city_count[city] += 1
	
	# Probability of a city occuring in the tweets, stored in the form of a dict.
	priors = {}
	num_samples = len(labels)
	for city in cities:
		priors[city] = city_count[city] / num_samples
	
	
	# Bag of probabilities has the likelihood for words given city.
	bagOfProbs = bag.copy()
	for city in cities:
		for word in bagOfProbs[city].keys():
			# This is the P(Word|City)
			bagOfProbs[city][word] = bagOfProbs[city][word] / city_count[city]
	
	# Return the city count in the data, 
	return city_count,priors,bagOfProbs

# Classify the tweets given the tweets, the priors for the cities and the likelihoods of words given cities.
def classifyTweet(tweet,priors,likelihoods):
	city_probs = {}
	
	for city in cities:
		# Initially set the probability as the prior of the city.
		prob = priors[city]
		for word in tweet:
			# If the word has occured then multiply its likelihood.
			if word in likelihoods[city].keys():
				prob *= likelihoods[city][word]
				modified = True
			else:
				# Laplace smoothing explained initially.
				prob *= 0.0000001
		# Update the dictionary which stores the predicted probability for each city given the words.
		city_probs[city] = prob #* priors[city]
	probs_list = list(city_probs.values())
	max_prob_index = probs_list.index(max(probs_list))
	
	return list(city_probs.keys())[max_prob_index]
	
# Function to calculate the accuracy of the model, accuracy is the (number of correctly classified tweets)/(total number of tweets given to the model)
def calculateAccuracy(actual_labels,predicted_labels):
	count_success = 0
	for i in range(0,len(predicted_labels)):
		if actual_labels[i] == predicted_labels[i]:
			count_success += 1
	return count_success/len(predicted_labels)

# Function which gives top words for a city.
def topWords():
	for city in cities:
		print(city)
		sorted_likelihood = sorted(likelihoods[city].items(), key=operator.itemgetter(1))
		print(sorted_likelihood[-5:],"\n")

# Start of execution

# Command line arguments
train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# Training phase
# Parse the training data set as labels and tweets.	
training_labels, training_tweets = parseTweets(train_file)

# Get the bag of words model for the training tweets.
training_bag = bagOfWordsModel(training_labels,training_tweets)

# Get the city count, priors and likelihoods for the training data 
city_count,priors,likelihoods = computeProbabilities(training_bag,training_labels)

# Classify the training data using the priors and the likelihoods.
predicted_labels = []

for i in range(len(training_tweets)):
	predicted_city = classifyTweet(training_tweets[i],priors,likelihoods)
	predicted_labels.append(predicted_city)

print("Training accuracy = ",calculateAccuracy(training_labels,predicted_labels)*100,"%")

# Testing phase
# Parse the training data set as labels and tweets.	
testing_labels, testing_tweets = parseTweets(test_file)

# Split the testing tweets into words and remove stop words.
for line in testing_tweets:
	line = [w for w in line if not w in stop_words]

# Clear the list used for storing predicted labels.
predicted_labels.clear()

# Classify the testing tweets using the learnt priors and the likelihoods.
for i in range(len(testing_tweets)):
	predicted_city = classifyTweet(testing_tweets[i],priors,likelihoods)
	predicted_labels.append(predicted_city)
	
print("Testing accuracy = ",calculateAccuracy(testing_labels,predicted_labels)*100,"%")

# Print the top 5 words for each city
topWords()

# Output to file.
with open(output_file,'w+') as out_file:
	for i in range(len(predicted_labels)):
		out_file.write(predicted_labels[i]+' '+testing_labels[i]+' '+' '.join(testing_tweets[i])+'\n')