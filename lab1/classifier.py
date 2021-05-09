import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import collections

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set

    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the
    smoothed estimates of q_d
    """
    ### TODO: Write your code here

    #get word frequncies in each email category
    #key:word, value: number of occurences in this email loader
    spam_dict = util.get_word_freq(file_lists_by_category[0])
    ham_dict = util.get_word_freq(file_lists_by_category[1])

    #get total length of each email loader
    spam_length = sum(spam_dict.values())
    ham_length = sum(ham_dict.values())

    #get the length of the dictionary: D
    dict_D = util.Counter()
    for key in spam_dict:
        dict_D[key] += spam_dict[key]
    for key in ham_dict:
        dict_D[key] += ham_dict[key]
    D = len(dict_D)

    spam_distribution = {}
    ham_distribution = {}
    #get the distributions of two email loaders
    for i in dict_D:
        spam_distribution[i] = (spam_dict[i] + 1) / (D + spam_length)

    for i in dict_D:
        ham_distribution[i] = (ham_dict[i] + 1) / (D + ham_length)
    #create the required tuple
    probabilities_by_category = (spam_distribution, ham_distribution)
    return probabilities_by_category


def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)],
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    spam_distribution = 0
    ham_distribution = 0
    word_frequency = util.get_word_freq([filename])
    for w in word_frequency:
        if w in probabilities_by_category[0]:
             spam_distribution += word_frequency[w] * np.log(probabilities_by_category[0][w])
        if w in probabilities_by_category[1]:
             ham_distribution += word_frequency[w] * np.log(probabilities_by_category[1][w])
    spam_distribution += np.log(prior_by_category[0])
    ham_distribution += np.log(prior_by_category[1])

    predict = ""
    if(spam_distribution > ham_distribution):
        predict = "spam"
    else:
        predict = "ham"

    word_distribution = [spam_distribution, ham_distribution]

    classify_result = (predict, word_distribution)

    return classify_result

if __name__ == '__main__':

    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))


    # Learn the distributions
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))


    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    print("----type 1 and 2 here-----")
    offset = [-1E2, -1E1, -1E0, 1E0, 1E1]
    type1 = []
    type2 = []
    for offset_value in offset:
        performance_measures = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category)

            #add offset
            if(log_posterior[0] + offset_value > log_posterior[1]):
                label = "spam"
            else:
                label = "ham"

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        type1.append(performance_measures[0][1])
        type2.append(performance_measures[1][0])

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
    plt.title("Type1 vs Type2 Error")
    for i in range(0, len(type1)):
        plt.scatter(type1[i], type2[i])

    plt.xlabel("type1")
    plt.ylabel("type2")
    plt.legend(offset, loc='best')
    plt.show()
