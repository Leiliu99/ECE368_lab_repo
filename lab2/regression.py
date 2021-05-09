import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    # (1) mu and cov
    mu_a = np.array([0, 0])
    cov_a = np.array([[beta, 0], [0, beta]])
    # (2) get contour
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)

    x_set = np.concatenate((X_flat, Y_flat), axis=1)
    Pa = util.density_Gaussian(mu_a, cov_a, x_set)
    Pa = Pa.reshape((20, 20))  # reshape back to 2D

    plt.contour(X, Y, Pa)

    # (3) draw the true value of a
    plt.scatter([-0.1], [-0.5], color='red')

    # (4) add title, labels
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('P(a)')

    plt.savefig('prior.pdf')
    plt.show()
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    #(1)get posterior size
    posterior_size = x.shape[0]

    # (2) mu and cov of posterior
    cov_a_inv = np.array([[1/beta, 0], [0, 1/beta]])
    add_one = np.ones((posterior_size, 1), dtype=int)
    x_add_one = np.concatenate((add_one, x), axis = 1)

    Cov = np.linalg.inv(cov_a_inv + 1/sigma2 * np.matmul(np.transpose(x_add_one), x_add_one))
    mu = np.matmul(np.linalg.inv(sigma2 * cov_a_inv + np.matmul(np.transpose(x_add_one), x_add_one)), np.matmul(np.transpose(x_add_one), z))

    # (3) get contour
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    X_flat = X_flat.reshape(-1, 1)
    Y_flat = Y_flat.reshape(-1, 1)

    x_set = np.concatenate((X_flat, Y_flat), axis=1)
    posterior = util.density_Gaussian(np.transpose(mu), Cov, x_set)
    posterior = posterior.reshape((X.shape[0], Y.shape[0]))  # reshape back to 2D

    plt.contour(X, Y, posterior)

    # (3) draw the true value of a
    plt.scatter([-0.1], [-0.5], color='red')

    # (4) add title, labels
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('posterior' + str(posterior_size))

    plt.savefig('posterior' + str(posterior_size) + '.pdf')
    plt.show()
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    # (1)get train size
    train_size = x_train.shape[0]

    # (2)get mu and cov of prediction
    x = np.asarray(x)
    x = x.reshape(x.shape[0], 1)
    add_one = np.ones((x.shape[0], 1))
    x_add_one = np.concatenate((add_one, x), axis=1)

    mu_prediction = np.matmul(x_add_one, mu)
    cov_prediction = np.matmul(np.matmul(x_add_one, Cov), np.transpose(x_add_one)) + sigma2

    # (3) get plot
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.scatter(x_train, z_train, color='red')
    plt.errorbar(x = x, y = mu_prediction, yerr = np.sqrt(np.diagonal(cov_prediction)), color = 'blue')

    plt.xlabel("input")
    plt.ylabel("prediction")
    plt.legend(['training samples', 'prediction range'])
    plt.title("prediction based on " + str(train_size) + " training samples")
    plt.savefig('predict' + str(train_size) + '.pdf')
    plt.show()

    return

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1

    # prior distribution p(a)
    priorDistribution(beta)
    
    # number of training samples used to compute posterior
    ns = [1, 5, 100]
    for i in range(len(ns)):
        # used samples
        x = x_train[0:ns[i]]
        z = z_train[0:ns[i]]

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)

        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    
