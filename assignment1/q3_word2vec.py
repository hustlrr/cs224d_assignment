# coding=utf-8

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / xn
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   

    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         

    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               

    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  

    ### YOUR CODE HERE
    y_pred = softmax(np.dot(outputVectors, predicted))  # outputVectors是context words的向量(W,d),predicted是中心词的向量表示(1,d)
    if np.isclose(np.sum(y_pred), 1) == False:
        print(outputVectors, predicted)
        raise Exception
    cost = -np.log(y_pred[target])  # 其他维y=0

    z = y_pred[:]  # (W,1)
    z[target] -= 1
    gradPred = np.dot(outputVectors.T, z)  # (d,1)
    grad = np.outer(z, predicted)   # (W, d)
    ### END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    gradPred = np.zeros_like(predicted)
    grad = np.zeros_like(outputVectors)

    z = sigmoid(np.dot(outputVectors[target], predicted))  # outputVectors(W,d),predicted(1,d),(W,1)

    # 首先计算根据正确的单词更新
    cost -= np.log(z)
    gradPred += ((z-1.) * outputVectors[target])  # (1,d)
    grad[target] += ((z - 1.) * predicted)
    # 根据负采样生成的词语更新
    for i in range(K):
        idx = dataset.sampleTokenIdx()
        z = sigmoid(np.dot(outputVectors[idx], predicted))
        cost -= np.log(1 - z)
        gradPred += (z * outputVectors[idx])  # (1,d)
        grad[idx] += (z * predicted)  # (W,d)

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)

    current_word_idx = tokens[currentWord]
    vc = inputVectors[current_word_idx]  # predicted(d,1)
    for contextWord in contextWords:
        target_word_idx = tokens[contextWord]
        c_cost, gradPred, grad = word2vecCostAndGradient(predicted=vc, target=target_word_idx,
                                                         outputVectors=outputVectors,
                                                         dataset=dataset)
        # print 'target_word_idx = %s' % target_word_idx
        # print "Returning cost=%s gradPred=%s grad=%s" % (c_cost, gradPred, grad)
        cost += c_cost
        gradIn[current_word_idx] += gradPred
        gradOut += grad
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset,
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom
    return cost, grad


def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], [tokens[random.randint(0, 4)] \
                                              for i in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #                 dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                   dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
    #            negSamplingCostAndGradient)


if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()
