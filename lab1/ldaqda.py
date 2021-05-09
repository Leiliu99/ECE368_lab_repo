import numpy as np
import matplotlib.pyplot as plt
import util
from numpy.linalg import inv
from numpy.linalg import det

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models

    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA
    and 1 figure for QDA in this function
    """
    ### TODO: Write your code here

    #get total number of samples
    N = x.shape[0]

    #get partial set of people who are male and female
    male_set = x[y == 1]
    female_set = x[y == 2]


    #get mu
    mu = np.mean(x, 0)
    mu_male = np.mean(male_set, 0)
    mu_female = np.mean(female_set, 0)

    mu = mu.reshape(-1, 1)
    mu_male = mu_male.reshape(-1, 1)
    mu_female = mu_female.reshape(-1, 1)

    #get cov
    male_num = male_set.shape[0]
    female_num = female_set.shape[0]

    cov = np.zeros((2,2))
    cov_male = np.zeros((2, 2))
    cov_female = np.zeros((2, 2))

    for i in range(N):
        smaple = x[i]
        smaple = smaple.reshape(-1,1)
        cov += np.matmul(smaple - mu, np.transpose(smaple - mu))

    for i in range(male_num):
        smaple_male = male_set[i]
        smaple_male = smaple_male.reshape(-1, 1)
        cov_male += np.matmul(smaple_male - mu_male, np.transpose(smaple_male - mu_male))

    for i in range(female_num):
        smaple_female = female_set[i]
        smaple_female = smaple_female.reshape(-1, 1)
        cov_female += np.matmul(smaple_female - mu_female, np.transpose(smaple_female - mu_female))

    cov = cov / N
    cov_male = cov_male / male_num
    cov_female = cov_female / female_num



    return (mu_male,mu_female,cov,cov_male,cov_female)


def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis

    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    #total number of samples
    N = x.shape[0]
    #LDA
    inv_cov = inv(cov)
    beta_m_transpose = np.transpose(np.matmul(inv_cov,mu_male))
    gamma_m = (-1/2) * np.matmul(np.matmul(np.transpose(mu_male), inv_cov), mu_male)
    lda_male = np.matmul(beta_m_transpose, np.transpose(x)) + gamma_m

    beta_f_transpose = np.transpose(np.matmul(inv_cov, mu_female))
    gamma_f = (-1 / 2) * np.matmul(np.matmul(np.transpose(mu_female), inv_cov), mu_female)
    lda_female = np.matmul(beta_f_transpose, np.transpose(x)) + gamma_f

    #get mistake number
    lda_predict = lda_male - lda_female

    lda_predict[lda_predict > 0] = 1
    lda_predict[lda_predict < 0] = 2

    misclassified = (lda_predict != y)

    mis_lda = np.sum(misclassified) / N


    #qda
    inv_cov_m = inv(cov_male)
    inv_cov_f = inv(cov_female)
    det_cov_m = det(cov_male)
    det_cov_f = det(cov_female)

    qda_male = np.zeros((N, 1))
    qda_female = np.zeros((N,1))

    for i in range(N):
        x_val = x[i]
        x_val = x_val.reshape(-1,1)
        first_term_m = np.transpose(np.subtract(x_val,mu_male))
        first_term_m = np.matmul(first_term_m, inv_cov_m)
        first_term_m = np.matmul(first_term_m, np.subtract(x_val,mu_male))
        first_term_m = (-1/2) * first_term_m

        second_term_m = (-1/2) * np.log(det_cov_m)

        qda_male[i] = first_term_m + second_term_m

        first_term_f = np.transpose(np.subtract(x_val, mu_female))
        first_term_f = np.matmul(first_term_f, inv_cov_f)
        first_term_f = np.matmul(first_term_f, np.subtract(x_val, mu_female))
        first_term_f = (-1 / 2) * first_term_f

        second_term_f = (-1 / 2) * np.log(det_cov_f)

        qda_female[i] = first_term_f + second_term_f

    # get mistake number
    qda_predict = qda_male - qda_female

    qda_predict[qda_predict > 0] = 1
    qda_predict[qda_predict < 0] = 2

    misclassified = (np.transpose(qda_predict) != y)

    mis_qda = np.sum(misclassified) / N



    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)


    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)

    print(mis_LDA)
    print(mis_QDA)


    ##plot N colored data points
    male_set = x_train[y_train == 1]
    female_set = x_train[y_train == 2]
    male_num = male_set.shape[0]
    female_num = female_set.shape[0]
    #LDA
    plt.figure(figsize = (10,10))
    plt.xlim([50, 80])
    plt.ylim([80, 280])
    #(1) plot dots
    plt.scatter(male_set[:male_num, 0], male_set[:male_num, 1], color = 'blue')
    plt.scatter(female_set[:female_num, 0], female_set[:female_num, 1], color='red')
    #(2) get contour
    x = np.arange(50, 80, 0.5)
    y = np.arange(80, 280, 0.5)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)

    x_set = np.concatenate((X_flat, Y_flat), axis=1)
    # print(x_set.shape)
    Z_male = util.density_Gaussian(np.transpose(mu_male),cov,x_set)
    Z_female = util.density_Gaussian(np.transpose(mu_female),cov,x_set)

    Z_male = Z_male.reshape((400, 60)) #reshape back to 2D
    Z_female = Z_female.reshape((400, 60))

    plt.contour(X, Y, Z_male, colors = 'blue')
    plt.contour(X, Y, Z_female, colors='red')

    #(3) plot boundary
    inv_cov = inv(cov)
    beta_m_transpose = np.transpose(np.matmul(inv_cov, mu_male))
    gamma_m = (-1 / 2) * np.matmul(np.matmul(np.transpose(mu_male), inv_cov), mu_male)
    lda_male = np.matmul(beta_m_transpose, np.transpose(x_set)) + gamma_m

    beta_f_transpose = np.transpose(np.matmul(inv_cov, mu_female))
    gamma_f = (-1 / 2) * np.matmul(np.matmul(np.transpose(mu_female), inv_cov), mu_female)
    lda_female = np.matmul(beta_f_transpose, np.transpose(x_set)) + gamma_f

    # get boundary
    lda_predict = lda_male - lda_female
    # print(lda_predict.shape)
    lda_predict = lda_predict.reshape((400,60))#reshape back to 2D

    plt.contour(X, Y, lda_predict, 0)#boundary with specified level lda_predict = 0

    #add title, labels
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.legend(['male', 'female'])
    plt.title('LDA visualization')

    plt.show()

    #QDA
    plt.figure(figsize=(10, 10))
    plt.xlim([50, 80])
    plt.ylim([80, 280])
    # (1) plot dots
    plt.scatter(male_set[:male_num, 0], male_set[:male_num, 1], color='blue')
    plt.scatter(female_set[:female_num, 0], female_set[:female_num, 1], color='red')
    # (2) get contour

    plt.contour(X, Y, Z_male, colors='blue')
    plt.contour(X, Y, Z_female, colors='red')
    # (3) plot boundary
    inv_cov_m = inv(cov_male)
    inv_cov_f = inv(cov_female)
    det_cov_m = det(cov_male)
    det_cov_f = det(cov_female)
    qda_male = np.zeros((x_set.shape[0], 1))
    qda_female = np.zeros((x_set.shape[0], 1))

    for i in range(x_set.shape[0]):
        x_val = x_set[i]
        x_val = x_val.reshape(-1, 1)
        first_term_m = np.transpose(np.subtract(x_val, mu_male))
        first_term_m = np.matmul(first_term_m, inv_cov_m)
        first_term_m = np.matmul(first_term_m, np.subtract(x_val, mu_male))
        first_term_m = (-1 / 2) * first_term_m

        second_term_m = (-1 / 2) * np.log(det_cov_m)

        qda_male[i] = first_term_m + second_term_m

        first_term_f = np.transpose(np.subtract(x_val, mu_female))
        first_term_f = np.matmul(first_term_f, inv_cov_f)
        first_term_f = np.matmul(first_term_f, np.subtract(x_val, mu_female))
        first_term_f = (-1 / 2) * first_term_f

        second_term_f = (-1 / 2) * np.log(det_cov_f)

        qda_female[i] = first_term_f + second_term_f

    # get boundary
    qda_predict = qda_male - qda_female
    qda_predict = qda_predict.reshape((400, 60))
    plt.contour(X, Y, qda_predict, 0)  # boundary with specified level qda_predict = 0

    # add title, labels
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.legend(['male', 'female'])
    plt.title('QDA visualization')

    plt.show()

    

    
    
    

    
