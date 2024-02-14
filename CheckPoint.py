import torch
# CALLBACKS SAVE

def save_callbacks(epoch,optimizer,model,callback_path):
    print("Callback is saving...")
    
    cALLBACK={"epoch":epoch,
              "optimizer_state":optimizer.state_dict(),
              "model_state": model.state_dict()}
    
    torch.save(cALLBACK,f=callback_path)
    
# CALLBACK LOADING
def load_callbacks(callback,optimizer,Model):
    print("CALLBACKS ARE LOADING...")
    
    starting_epoch=callback["epoch"]
    optimizer.load_state_dict(callback["optimizer_state"])
    Model.load_state_dict(callback["model_state"])
    return starting_epoch

    