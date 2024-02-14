import torch,model,config,dataset,Training,Testing,CheckPoint,prediction
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")



# Create SummaryWriter for Tensorboard Items that We Will Visualize
Tesnsorboard_writer=SummaryWriter(log_dir=config.TENSORBOARD_WRITER_PATH)


# Control cuda
if torch.cuda.is_available():
    devices="cuda"  
else:
    devices="cpu" 
    
    
# Create Transformer
transformer=dataset.transformer(resize_img_size=config.RESHAPE_IMG_SIZE)

# Loading path of datasets
class_names_dict,train_dataset_paths=dataset.loading_datasets_path(train_path=config.TRAIN_PATH)


# Create train and test dataset
train_dataset=dataset.Dataset(train_dataset_paths,classes_dict=class_names_dict,transformer=transformer)


# Random split for validation dataset
valid_dataset,train_dataset,test_dataset=dataset.random_split_fn(train_dataset=train_dataset)


# Create Dataloaders with batch_size
train_dataloader,valid_dataloader,test_dataloader=dataset.dataloader(train_dataset=train_dataset,
                                                                     valid_dataset=valid_dataset,
                                                                     test_dataset=test_dataset,
                                                                     BATCH_SIZE=config.BATCH_SIZE)


# Embedding Images to Tensorboard
dataset.Embedding_Image_Tensorboard(dataloader=train_dataloader,devices=devices,tensorboard_writer=Tesnsorboard_writer)
    

# Create Model
Model,optimizer,loss_fn=model.create_model(label_size=len(class_names_dict),
                                           initial_size=config.CHANNEL_SIZE,
                                           devices=devices,
                                           learning_rate=config.LEARNING_RATE,
                                           )


# Loading callbacks
if config.LOAD==True:

    callback=torch.load(config.CALLBACKS_PATH)
    starting_epoch=CheckPoint.load_callbacks(callback=callback,
                                             optimizer=optimizer,
                                             Model=Model)

else:
    starting_epoch=0
    print("Training is starting from scratch...")
    



# Training
if config.TRAIN==True:
    Training.Train_Model(Model=Model,
                EPOCHS=config.EPOCH,
                starting_epoch=starting_epoch,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                save_callbacks=CheckPoint.save_callbacks,
                devices=devices,
                callback_path=config.CALLBACKS_PATH,
                Tensorwriter=Tesnsorboard_writer)


#Test Model
if config.TEST_STEP == True:
    Test_Result_Dict,test_pred_list,test_label_list=Testing.Test_Model(test_dataloader=test_dataloader,
                                                                          Model=Model,
                                                                          loss_fn=loss_fn,
                                                                          devices=devices,
                                                                          VISUALIZATION_TEST=True)

# Prediction
if config.PREDICTION==True:
    prediction.Prediction(Model=Model,
                          prediction_folder_path=config.PREDICTION_PATH,
                          idx_to_label=class_names_dict,
                          reshape_img_size=config.RESHAPE_IMG_SIZE)




