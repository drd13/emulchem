import pickle
import numpy as np
from emulchem.network import NeuralNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


#rad_net = NeuralNet(input_size=1,hidden_size=200,hidden_size2=100,hidden_si    ze3=50,num_outputs=1)


class Emulator():
    def __init__(self,specie):
        self.specie = specie

    def get_prediction_list(self,x):
        x = np.array(x)
        x_val = self.input_scaler.transform(x.reshape(1,-1))
        y_val = self.neural_network(torch.FloatTensor(x_val)).detach().numpy()[0]
        y_val = self.output_scaler.inverse_transform(y_val.reshape(1,-1))
        return y_val[0][0]        




class ChemistryEmulator(Emulator):
    def __init__(self,specie):
        Emulator.__init__(self,specie)
        self.neural_network = NeuralNet(input_size=6,hidden_size=200,hidden_size2=100,hidden_size3=50,num_outputs=1)
        self.neural_network.load_state_dict(torch.load("/home/drd13/Project/analysis/models/network{}".format(self.specie)))
        self.input_scaler = pickle.load(open("/home/drd13/Project/analysis/models/minMaxScaler.p", "rb"))
        self.output_scaler = pickle.load(open("/home/drd13/Project/analysis/models/scaler{}.p".format(self.specie),"rb" ))
    
    def get_prediction(self,radfield,zeta,density,av,temperature,metallicity):
        x = [radfield,zeta,density,av,temperature,metallicity]
        y = np.exp(self.get_prediction_list(x))
        return y

class RadexEmulator(Emulator):
    def __init__(self,specie,transition):
        Emulator.__init__(self,specie)
        self.transition  = transition
        self.neural_network = NeuralNet(input_size=3,hidden_size=200,hidden_size2=100,hidden_size3=50,num_outputs=1)
        self.neural_network.load_state_dict(torch.load("/home/drd13/Project/analysis/modelsRadex/network{}{}".format(self.specie,self.transition)))
        self.input_scaler = pickle.load(open("/home/drd13/Project/analysis/modelsRadex/minMaxScaler{}{}.p".format(self.specie,self.transition), "rb"))
        self.output_scaler = pickle.load(open("/home/drd13/Project/analysis/modelsRadex/scaler{}{}.p".format(self.specie,self.transition),"rb" ))

    def get_prediction(self,temperature,density,column_density,line_width=1):
        """Temperature:Kelvin
        Density:...
        column_density:
        line-width:
        """
        x = [temperature,density,np.log10(column_density/line_width)]
        #y = np.power(10,self.get_prediction_list(x))*line_width
        y = self.get_prediction_list(x)*line_width
        return y



    


