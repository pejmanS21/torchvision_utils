import torch

def isnotebook():
    """check if code run in notebook

    Returns:
        bool: True= Jupyter notebook or Colab 
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or 'Shell':
            return True   # Jupyter notebook or qtconsole or Colab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm.notebook import tqdm
else: 
    from tqdm import tqdm




def train(epoch, n_epochs, model, dl, loss_func, device, optimizer, len_ds: int):
    """train model

    Args:
        epoch (int): for tqdm info
        n_epochs (int): for tqdm info
        model
        dl (pytorch dataloader): 
        loss_func 
        device ('str'): "cuda" or "cpu"
        optimizer 
        len_ds (int): for tqdm info

    Returns:
        Tuple: epoch loss and accuracy
    """
    model.train(True)
    torch.set_grad_enabled(True)
    
    epoch_loss = 0.0
    epochs_acc = 0
    
    tq_batch = tqdm(dl, total=len(dl))
    for images, labels in tq_batch:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outs = model(images)
        _, preds = torch.max(outs, 1)
        
        loss = loss_func(outs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epochs_acc += torch.sum(preds == labels).item()
        
        tq_batch.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        tq_batch.set_postfix_str('loss = {:.4f}'.format(loss.item()))

            
    epoch_loss = epoch_loss / len(dl)
    epochs_acc = epochs_acc / len_ds

    return epoch_loss, epochs_acc


def evaluate(model, dl, loss_func, device, len_val_ds: int):
    """evaluate model

    Args:
        model 
        dl (pytorch dataloader): 
        loss_func 
        device ('str'): "cuda" or "cpu"

        len_val_ds (int): for tqdm info

    Returns:
        Tuple: epoch loss and accuracy for validation set
    """

    model.train(False)

    epoch_loss = 0
    epochs_acc = 0
    tq_batch = tqdm(dl, total=len(dl), leave=False)
    for images, labels in tq_batch:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = loss_func(outputs, labels)

        epoch_loss += loss.item()
        epochs_acc += torch.sum(preds == labels).item()
        tq_batch.set_description(f'Evaluate Model')
        
    epoch_loss = epoch_loss / len(dl)
    epochs_acc = epochs_acc / len_val_ds

    return epoch_loss, epochs_acc


def fit(n_epochs, model, train_dataloader, valid_dataloader, loss_func, device, optimizer):
    """fit model (train and evaluate)

    Args:
        n_epochs (int): for info
        model (pytorch model)
        train_dataloader (pytorch dataloader)
        valid_dataloader (pytorch dataloader)
        loss_func 
        device ('str'): 'cuda' or 'cpu
        optimizer 
    Returns:
        history (list of dict): for visualize model results
    """
    
    history = []
    val_loss_ref = float('inf')
    patient = 5
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        loss, acc = train(epoch, n_epochs, model, train_dataloader, loss_func, device, optimizer)
        
        torch.cuda.empty_cache()
        val_loss, val_acc = evaluate(model, valid_dataloader, loss_func, device)
        
        history.append({'loss': loss, 'acc': acc, 'val_loss': val_loss, 'val_acc': val_acc})

        statement = "[loss]={:.4f} - [acc]={:.4f} - \
[val_loss]={:.4f} - [val_acc]={:.4f}".format(loss, acc, val_loss, val_acc,)
        print(statement)
        ####### Checkpoint
        if val_loss < val_loss_ref:
            patient = 5
            val_loss_ref = val_loss
            model_path = './Face_Recognition_checkpoint.pth'
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saving model dict, Epoch={epoch + 1}")
        else:
            if patient == 0: 
                break
            print(f"[INFO] {patient} lives left!")
            patient -= 1
            

    return history