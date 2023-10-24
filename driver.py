
#%%

from data.celeb_utils import get_dataset
from pathlib import Path    
from torch.utils.data import DataLoader
from model.vit import ViTConfig, ViTModel
import torch
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import classification_report


data_path = Path('/shared/datasets/Celeb-A/')
device = torch.device('cuda:0')


tr_dataset = get_dataset(data_path, split='train')
te_dataset = get_dataset(data_path, split='test')

tr_dataloader = DataLoader(tr_dataset, batch_size=386, shuffle=True)
te_dataloader = DataLoader(te_dataset, batch_size=4096, shuffle=True)

config = ViTConfig()
model = ViTModel(config).to(torch.bfloat16).to(device)
optimizer = AdamW (model.parameters(), lr=5e-5)



def eval(model, te_dataloader):
    model.eval()

    predictions = []
    targets = []
    losses = []

    for images, target in te_dataloader:
        images = images.to(torch.bfloat16).to(device)
        target = target.to(torch.bfloat16).to(device)
        with torch.no_grad():
            logits, loss = model(images, target)
        
        preds = torch.sigmoid(logits.detach())
        # correct = target == (probs > 0.5).to(torch.int64)
        
        predictions.append(preds.cpu().to(torch.float32).numpy())
        losses.append(loss.detach().cpu().item())
        targets.append(target.cpu().to(torch.float32).numpy())


    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    report = classification_report(targets.ravel(), (predictions.ravel() > 0.5).astype(int), output_dict=True)
    macro_avg = report['macro avg']
    acc = report['accuracy']
    log = f"P={macro_avg['precision']:.3f} R={macro_avg['recall']:.3f} F1={macro_avg['f1-score']:.3f} ACC={acc:.3f}, end=' '"
    print(log)
    return losses, report

tr_losses = []
te_losses = []

#%%


for ix, (images, labels) in enumerate(tr_dataloader):
    model.train()
    optimizer.zero_grad()
    images = images.to(torch.bfloat16).to(device)
    labels = labels.to(torch.bfloat16).to(device)

    logits, loss = model(images, labels)
    loss.backward()
    optimizer.step()

    if (ix % 10) == 0:
        tr_loss = loss.detach().item()
        print(ix, 'avg train loss:', tr_loss)
        tr_losses.append(tr_loss)
    
    if (ix % 100) == 0:
        losses, report = eval(model, te_dataloader)
        avg_loss = sum(losses) / len(losses)
        te_losses.append(avg_loss)
        print('LOSS:', avg_loss)
        print()

 #%%        

#%%







# %%
