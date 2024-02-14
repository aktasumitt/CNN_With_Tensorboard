from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import glob,tqdm



def transformer(resize_img_size):
    "If you want u can use data augmentation. I didnt use because training will take too much time if i used"
    
    transformer=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                                transforms.Resize((resize_img_size,resize_img_size)),
                                # transforms.RandomHorizontalFlip(0.5),
                                # transforms.RandomRotation(degrees=30),
                                # transforms.RandomAutocontrast(0.3)
                                ]) 
     
    
    return transformer


def loading_datasets_path(train_path):
    
    class_names_dict={}
    train_dataset_paths=[]
    
    for file in glob.glob(pathname=train_path+"/*"):
        
        class_name=file.split("\\")[-1]
        class_names_dict[len(class_names_dict)]=class_name
        
        for img in glob.glob(file+"/*"):
            train_dataset_paths.append(img)
    
    return class_names_dict,train_dataset_paths



class Dataset():
    def __init__(self,data_path_list,classes_dict,transformer) :
        
        self.transformer=transformer
        self.path=data_path_list
        self.classes_dict=classes_dict
     
    def __len__(self):
        return len(self.path)
    
    def __get_img_class__(self,index):
        return self.classes_dict[index]
    
    def __getitem__(self,index):
        
        label=-1
        image=Image.open(self.path[index]).convert("RGB")
        image=self.transformer(image)
        
        for i in range(len(self.classes_dict)):
            if self.classes_dict[i] in self.path[index]:
                label=i
                    
        return (image,label)
        

def random_split_fn(train_dataset):
    valid_size=int(len(train_dataset)*0.2)
    train_size=len(train_dataset)-(valid_size*2)
    valid_dataset,train_dataset,test_dataset=random_split(train_dataset,lengths=[valid_size,train_size,valid_size])

    return valid_dataset,train_dataset,test_dataset



def dataloader(train_dataset,valid_dataset,test_dataset,BATCH_SIZE):
    
    train_dataloader=DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)

    valid_dataloader=DataLoader(valid_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)

    test_dataloader=DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            drop_last=True)
    
    print("Dataloaders Are Created.\n")

    return train_dataloader,valid_dataloader,test_dataloader



def Embedding_Image_Tensorboard(dataloader,devices,tensorboard_writer):
    pb=tqdm.tqdm(range(10),"Embedding img on Tensorboard")
    
    for batch,(img,label) in enumerate(dataloader,0):
        img=img.to(devices)
        img_grid=make_grid(img,nrow=15)
        tensorboard_writer.add_image("Train_Image",img_grid,global_step=batch)
        pb.update(1)
        if batch==10:
            break
    pb.close()
        
        
