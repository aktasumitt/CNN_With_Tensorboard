import torch,tqdm

def Train_Model(Model,EPOCHS,starting_epoch,devices,train_dataloader,valid_dataloader,optimizer,loss_fn,save_callbacks,callback_path,Tensorwriter):

    writer_step=1

    for epoch in range(starting_epoch,EPOCHS):
        print(f"\n{epoch+1}.epoch is starting...\n")
        
        loss_train=0.0
        train_total=0.0
        train_correct=0.0
        
        prog_bar=tqdm.tqdm(range(len(train_dataloader)),"Training Progress")
        
        for batch,(data_train,labels_train) in enumerate(train_dataloader,0):
            
            data_train=data_train.to(devices)
            labels_train=labels_train.to(devices)
            
            optimizer.zero_grad()
            out_train=Model(data_train)
            _,predict_train=torch.max(out_train,1)
            loss_t=loss_fn(out_train,labels_train)
            loss_t.backward()
            optimizer.step()

            loss_train+=loss_t.item()
            train_total+=labels_train.size(0)
            train_correct+=(predict_train==labels_train).sum().item()
            
            if batch%40==39:
                with torch.no_grad():
                    loss_valid=0.0
                    correct_valid=0.0
                    total_valid=0.0
                    
                    for batch_valid,(data_valid,label_valid) in enumerate(valid_dataloader,0):
                        
                        data_valid=data_valid.to(devices)
                        label_valid=label_valid.to(devices)
                        out_valid=Model(data_valid)
                        _,predict_valid=torch.max(out_valid,1)
                        loss_v=loss_fn(out_valid,label_valid)
                        
                        loss_valid+=loss_v.item()
                        correct_valid+=(predict_valid==label_valid).sum().item()
                        total_valid+=label_valid.size(0)

                # Create Tensorboard Writer with adding training item
                Tensorwriter.add_scalar("Train_acc",100*(train_correct/train_total),global_step=writer_step)
                Tensorwriter.add_scalar("Valid_acc",100*(correct_valid/total_valid),global_step=writer_step)
                Tensorwriter.add_scalar("Train_Loss",(loss_train/batch+1),global_step=writer_step)
                Tensorwriter.add_scalar("Valid_Loss",(loss_valid/batch_valid+1),global_step=writer_step)
                writer_step+=1
                
                prog_bar.set_postfix_str(f"epoch:{epoch+1}/{EPOCHS}"
                                        f"   Batch:{batch+1}/{len(train_dataloader)}"
                                        f"   Train_accuracy:{100*(train_correct/train_total):.3f}"
                                        f"   Train_loss:{(loss_train/batch+1):.3f}"
                                        f"   Valid_accuracy:{100*(correct_valid/total_valid):.3f}"
                                        f"   Valid_loss:{(loss_valid/batch_valid+1):.3f}"
                                        )
            prog_bar.update(1)
        
        prog_bar.close() 
        
        # Save Callbacks
        save_callbacks(epoch=epoch,optimizer=optimizer,model=Model,callback_path=callback_path) 
        
        