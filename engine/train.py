import argparse, os
import torch
from torch import nn
from dataset import get_dataloader
from models import create_model
import matplotlib.pylab as plt 


def train_model(model, train_loader, val_loader, model_path, num_epochs=50, lr=0.001, patience=10, restart_training=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if restart_training:
        weights = f'{model_path}/best_weights.pt'
        if os.path.exists(weights):
            model.load_stat_dict(torch.load(weights))
            print(f'Model training resumed from weight: {weights}')
        else:
            print(f'Model path {weights} not exist. The training will resume from scartch')
   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history = {'train_loss': [],
               'valid_loss':[],
    }
   

    best_score = 0.0
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
       
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs, _ = model(x1, x2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()
       
        # Validation
        model.eval()
        val_loss = 0.0
       
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                outputs = model(x1, x2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
               
        scheduler.step()

        ep_tr_loss = running_loss/len(train_loader)
        ep_vl_loss = val_loss/len(val_loader)

        history['train_loss'].append(ep_tr_loss)
        history['valid_loss'].append(ep_vl_loss)
       
       
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {ep_tr_loss:.4f}, '
              f'Val Loss: {ep_vl_loss:.4f}, ')
        
        if epoch == 1:
            best_score = ep_vl_loss
        elif epoch>=1:
            if ep_vl_loss <= best_score:
                best_score == ep_vl_loss
                counter = 0
                torch.save(model.get_state_dict(), f'{model_path}/best_weights.pt')
            else:
                counter+=1
            
            if counter >= patience:
                torch.save(model.get_state_dict(), f'{model_path}/final_weights.pt')
                plot_history(history=history, model_path=model_path)
                break
    torch.save(model.get_state_dict(), f'{model_path}/final_weights.pt')

    plot_history(history=history, model_path=model_path)

def plot_history(history, model_path):
    plt.plot(history['train_loss'], label='train-loss',c='r')
    plt.plot(history['valid_loss'], label='valid-loss', c='g')
    plt.legend()
    plt.savefig(f"{model_path}/history.png", dpi=350)
    plt.show()

def getargs():
    pars = argparse.ArgumentParser(description='take the arguments for the training')
    pars.add_argument('--data_dir', type=str, help='root data directory')
    pars.add_argument('--split_ratio', type=float, help='train validation data split ratio', default=0.8)
    pars.add_argument('--epochs', type=int, help='number of training epochs', default=50)
    pars.add_argument('--batch_size', type=int, help='batch size for model training', default=12)
    pars.add_argument('--lr', type=float, help='learning rate', default=0.001)
    pars.add_argument('--model_type', type=str, help='model type either resnet or combined')
    pars.add_argument('--num_class', type=int, help='number of damage classes')
    pars.add_argument('--pretrained', action='store_false')
    pars.add_argument('--resume', action='store_true', help='Whether to resume the training from existing weight')
    pars.add_argument('--resnet_depth', type=int, help='ResNet model depth either 50 or 101', default=50)
    pars.add_argument('--model_path', type=str, help='path to save the model', default='/')
    pars.add_argument('--patience', type=int, help='the number of epochs to wait the model not imroving', default=10)
    return pars.parse_args()

def main(args):

    train_loader, valid_loader = get_dataloader(data_dir=args.data_dir,
                                                split_ratio=argparse.split_ratio)

    model = create_model(num_classes=args.num_class,
                         model_type=args.model_type,
                         depth=args.resnet_depth,
                         pretrained=args.pretrained)

    train_model(model=model,
                train_loader=train_loader,
                val_loader=valid_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                model_path=args.model_path,
                patience=args.patience)