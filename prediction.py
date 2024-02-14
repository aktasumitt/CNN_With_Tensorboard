import torch,glob
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def Prediction(Model,prediction_folder_path,idx_to_label,reshape_img_size):
    
    with torch.no_grad():
        img_path_list=glob.glob(prediction_folder_path+"/*")
        
        for i,path in enumerate(img_path_list):
            img=Image.open(path)
            img_transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,),(0.5,)),
                                                transforms.Resize((reshape_img_size,reshape_img_size))])(img)
            Model.cpu()
            out=Model(img_transform.unsqueeze(0))
            _,pred=torch.max(out,1)
            
            print("Predict Letter: ",pred[0].item())
            
            if len(img_path_list)%3==0:
                plt.subplot(int(len(img_path_list)/3),3,i+1)
            
            elif len(img_path_list)%5==0:
                plt.subplot(int(len(img_path_list)/5),5,i+1)
            
            else: plt.subplot(int(len(img_path_list)/2),2,i+1)
            
            plt.imshow(torch.permute(img_transform,(1,2,0)))
            plt.xlabel("pred: "+idx_to_label[pred[0].item()])
            plt.xticks([])
            plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0.4,hspace=0.4)
            
        plt.show()
    