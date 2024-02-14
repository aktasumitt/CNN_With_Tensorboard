import torch,tqdm
import matplotlib.pyplot as plt

def Test_Model(test_dataloader,Model,loss_fn,devices,VISUALIZATION_TEST:bool=True):
        
        loss_test=0.0
        correct_test=0.0
        total_test=0.0
        
        pred_list=torch.zeros(0,dtype=torch.long,device="cpu")
        label_list=torch.zeros(0,dtype=torch.long,device="cpu")               
        prog_bar=tqdm.tqdm(range(len(test_dataloader)),"Testing Progress")
        
        for batch_test,(data_test,label_test) in enumerate(test_dataloader):
                       
                data_test=data_test.to(devices)
                label_test=label_test.to(devices)
                out_test=Model(data_test)
                _,predict_test=torch.max(out_test,1)
                loss_t=loss_fn(out_test,label_test)
                                
                loss_test+=loss_t.item()
                correct_test+=(predict_test==label_test).sum().item()
                total_test+=label_test.size(0)             
                accuracy_test=100*correct_test/total_test
                
                pred_list=torch.cat([pred_list,predict_test.view(-1).cpu()])
                label_list=torch.cat([label_list,label_test.view(-1).cpu()])
                
                prog_bar.set_postfix({"Test_Accuracy":accuracy_test,"Loss_Test":(loss_test/(batch_test+1))})
                prog_bar.update(1)
        
        random_indexes=torch.randint(0,5000,(20,))
        
        
        if VISUALIZATION_TEST==True:
                for i,index in enumerate(random_indexes):
                        
                        plt.subplot(5,4,i+1)
                        plt.imshow(torch.permute(test_dataloader.dataset[index][0],(1,2,0)))
                        plt.xticks([])
                        plt.xlabel(f"p={pred_list[index]},r={test_dataloader.dataset[index][1]}")
                
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
                
                plt.show()
                
        return {"Test_Accuracy":accuracy_test,"Loss_Test":(loss_test/batch_test)},pred_list,label_list