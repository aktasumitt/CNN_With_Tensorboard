import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self,label_size,initial_size) -> None:
        super(CNNModel,self).__init__()
        self.initial_size=initial_size
        self.label_size=label_size
        self.cnnlayers=nn.Sequential(nn.Conv2d(self.initial_size,out_channels=64,kernel_size=3), # (126,126)
                                     nn.BatchNorm2d(64),
                                     nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2,stride=2),# (63,63)
                                     
                                     nn.Conv2d(64,128,kernel_size=2),# (62,62)
                                     nn.BatchNorm2d(128),
                                     nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2,stride=2),#(31,31)
                                     
                                     nn.Conv2d(128,256,kernel_size=2),# (30,30)
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2,stride=2)# (15,15) 
                                     )
        
        self.linear_layers=nn.Sequential(nn.Linear(in_features=(256*15*15),out_features=1024),
                                         nn.ReLU(),
                                         nn.Linear(1024,512),
                                         nn.ReLU(),
                                         nn.Linear(512,128),
                                         nn.ReLU(),
                                         nn.Linear(128,label_size))
    def forward(self,data):
        x=self.cnnlayers(data)
        x=x.view(-1,256*15*15)
        out=self.linear_layers(x)  
        return out 


def create_model(label_size,initial_size,devices,learning_rate):
    
    #Create Model , optimizer and loss
    Model=CNNModel(label_size,initial_size)

    Model.to(device=devices)

    optimizer=torch.optim.Adam(params=Model.parameters(),lr=learning_rate)
    loss_fn=torch.nn.CrossEntropyLoss()
    
    print("Model is Created.\n")
    
    return Model,optimizer,loss_fn