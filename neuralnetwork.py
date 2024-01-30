import numpy as np





class neuralnetwork:

        
    def __init__(self, input_nodes, hidden_nodes, output_nodes=1): 
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.input_weights = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.hidden_weights = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5 # 1 by 3
        self.binary_loss = None



    def feed_forward(self, X):
        self.z1 = np.dot(self.input_weights, X) 
        self.a1 = self.sigmoid(self.z1) # 3 by 1
        self.z2 = np.dot(self.hidden_weights, self.a1) # 1 by 3 by 3 by 1
        out = self.sigmoid(self.z2)

        return out


    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    

    def backprop(self, inputs_list, targets_list):

        outputs = self.feed_forward(inputs_list)
        output_error = outputs - targets_list
        self.binary_loss = - np.mean(targets_list * np.log(outputs) + (1 - targets_list) * np.log(1 - outputs))
        sig_error = outputs * (1-outputs)
        self.dz2 = output_error * sig_error # 1 by 1
        self.dw2 = np.dot(self.dz2, self.a1.T) #1 by 1 by 1 by 3
        da = np.dot(self.hidden_weights.T, self.dz2)
        dz1 = da * self.a1 * (1-self.a1)
        self.dw1 = np.dot(dz1, inputs_list.T)

    def train(self, inputs_list, targets_list, epochs=100):

        for i in range(epochs):
            self.backprop(inputs_list, targets_list)
            print(self.binary_loss)
            self.hidden_weights = self.hidden_weights - (0.0001*self.dw2) # element wise vector substraction
            self.input_weights = self.input_weights - (0.0001*self.dw1)

        # updates the weights


















    


        


    
        
        


