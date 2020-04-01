import numpy as np

class Perceptron:
    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x) )


    # input dataset
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    # output dataset
    training_outputs = np.array([[0,1,1,0]]).T

    np.random.seed(1)

    random_weights = 2 * np.random.random((3,1))-1
    print("Welcome in perceptron network : ")
    print("random starting weights:")
    print("-----------------------")
    print(random_weights)

    def check(self) :
        for iterx in range(1):
            output = np.dot(self.training_inputs , self.random_weights)
            activiation_output = self.sigmoid(output)
            print("activiation_output:")
            print("-------------------")
            print(activiation_output)

result= Perceptron()
result.check()

