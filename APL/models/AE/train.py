from pathlib import Path
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import yaml
from .dataset import DASDataset
from .net import Net


def forward(model, device, loss_function, batch):
    x = Variable(batch).to(device=device, dtype=torch.float)
    xpred = model(x)
    loss = loss_function(x, xpred)
    return loss


def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def training_epoch(model, loader, device, loss_function, optimizer):
    total_loss = 0
    model.train()
    for batch in loader:
        loss = forward(model, device, loss_function, batch)
        backward(optimizer, loss)
        total_loss += loss.item()
    return total_loss/len(loader)


def validation_epoch(model, loader, device, loss_function):
    total_loss = 0
    model.eval()
    for batch in loader:
        loss = forward(model, device, loss_function, batch)
        total_loss += loss.item()
    return total_loss/len(loader)


def print_log(epoch, nepochs, train_loss, validation_loss):
    print('Epoch [{}/{}]: Training Loss: {:.6f}'.format(epoch+1, nepochs, train_loss))
    print('Epoch [{}/{}]: Validation Loss: {:.6f}'.format(epoch+1, nepochs, validation_loss))
    print('================================================')


def main(args):

    # Prepare paths
    path = Path(__file__).parent.resolve()
    Path(path/"../../../models").mkdir(parents=True, exist_ok=True)
    model_path = path/("../../../models/{net}_{train}_{dist_ini}_{dist_fin}".\
        format(net=args.net, train=args.train, dist_ini=args.dist_ini, dist_fin=args.dist_fin))

    # Read config files
    with open(path/("../../../configs/"+args.net+".yaml"), 'r') as f:
        net_cfg = yaml.load(f, yaml.FullLoader)
    with open(path/("../../../configs/"+args.train+".yaml"), 'r') as f:
        train_cfg = yaml.load(f, yaml.FullLoader)

    # Define model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(*net_cfg['net_design'].values()).to(device=device, dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])

    # Datasets for training and validation
    train_set = DASDataset('silence_train', *train_cfg['preprocessing'].values(), args.dist_ini, args.dist_fin)
    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=True)
    validation_set = DASDataset('silence_validation', *train_cfg['preprocessing'].values(), args.dist_ini, args.dist_fin)
    validation_loader = DataLoader(validation_set, batch_size=train_cfg['batch_size'], shuffle=True)

    #Train model
    loss_function = torch.nn.MSELoss()
    for epoch in range(train_cfg['nepochs']):
        train_loss = training_epoch(model, train_loader, device, loss_function, optimizer)
        validation_loss = validation_epoch(model, validation_loader, device, loss_function)
        print_log(epoch, train_cfg['nepochs'], train_loss, validation_loss)

    # Save model
    torch.save(model.state_dict(), model_path)