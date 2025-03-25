import torch
from architectures.ClassificationViT import ClassificationViT
import torch.optim as optim
from tqdm import tqdm, trange
from data.load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

if __name__ == '__main__':

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_data()

    # Initilize model (ClassificationViT)
    model = ClassificationViT(10)

    # Move model to GPU
    model.to(torch_device)

    # Create optimizer for the model
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-9)

    total_steps = 0
    num_epochs = 10
    train_logfreq = 100
    losses = []
    train_acc = []
    all_val_acc = []
    best_val_acc = 0

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            x, y = x.to(torch_device), y.to(torch_device)
            logits = model(x)
            loss = torch.mean(F.cross_entropy(logits, y))
            accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item(), train_acc=accuracy.item())

            if total_steps % train_logfreq == 0:
                losses.append(loss.item())
                train_acc.append(accuracy.item())

        # Validation
        val_acc = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(torch_device), y.to(torch_device)
            with torch.no_grad():
                logits = model(x)
            accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())
            val_acc.append(accuracy.item())
        model.train()

        all_val_acc.append(np.mean(val_acc))
        # Save the best model
        if np.mean(val_acc) > best_val_acc:
            best_val_acc = np.mean(val_acc)
        epoch_iterator.set_postfix(val_acc=np.mean(val_acc), best_val_acc=best_val_acc)

    plt.plot(losses)
    plt.title('Train Loss')
    plt.savefig("../reports/figures/train_loss.png", dpi=300, bbox_inches='tight')
    plt.figure()

    plt.plot(train_acc)
    plt.title('Train Accuracy')
    plt.savefig("../reports/figures/train_acc.png", dpi=300, bbox_inches='tight')
    plt.figure()

    plt.plot(all_val_acc)
    plt.title('Val Accuracy')
    plt.savefig("../reports/figures/val_acc.png", dpi=300, bbox_inches='tight')
