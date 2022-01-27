import h5py
import numpy as np
from pathlib import Path
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import yaml
from .dataset import DASDataset
from .net import Net
from ..utils import patches2data


def predict(model, test_loader, device):
    s_ise = []
    with torch.no_grad():
        for test_sample in test_loader:
            x = Variable(test_sample).to(device=device, dtype=torch.float)
            xpred = model(x)
            s_ise.extend(np.abs((x-xpred).numpy()[:,0,:,:]))
    return s_ise


def main(args):

    # Prepare paths
    path = Path(__file__).parent.resolve()
    Path(path/"../../../predictions").mkdir(parents=True, exist_ok=True)
    model_path = path/("../../../models/{net}_{train}_{dist_ini}_{dist_fin}".\
        format(net=args.net, train=args.train, dist_ini=args.dist_ini, dist_fin=args.dist_fin))
    pred_path = path/("../../../predictions/{net}_{train}_{predict}_{id}_{dist_ini}_{dist_fin}".\
        format(net=args.net, train=args.train, predict=args.predict, id=args.id,\
            dist_ini=args.dist_ini, dist_fin=args.dist_fin))

    # Read config files
    with open(path/("../../../configs/"+args.net+".yaml"), 'r') as f:
        net_cfg = yaml.load(f, yaml.FullLoader)
    with open(path/("../../../configs/"+args.predict+".yaml"), 'r') as f:
        predict_cfg = yaml.load(f, yaml.FullLoader)

    # Load Model
    device = 'cpu'
    model = Net(*net_cfg['net_design'].values()).to(device=device, dtype=torch.float).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Test Loader
    test_set = DASDataset(args.id, *predict_cfg.values(), args.dist_ini, args.dist_fin)
    test_loader = DataLoader(test_set, batch_size=2048, shuffle=False)

    # Make Predictions
    s_ise = predict(model, test_loader, device)

    # Save Predictions
    s_ise = patches2data(np.array(s_ise), predict_cfg['mt'], test_set.Lt, predict_cfg['md'],test_set.Ld)*test_set.std
    with h5py.File(pred_path,'w') as hf:
        hf.create_dataset('s_ise', data=s_ise)