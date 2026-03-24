import torch
import os
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from dataset.omg_dataset import OMGDataset, collate
import torch.optim as optim
from mead_temporal_transformer import TransformerVA
import matplotlib.pyplot as plt

BATCH, EPOCHS, LR, FEATURE_DIM, = 4, 20, 1e-4, 1584

ds = OMGDataset(os.path.join(os.path.dirname(__file__), "dataset/train_metadata_fixed.json"))

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size

generator = torch.Generator().manual_seed(42)
train_data, test_data = random_split(ds, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=True, collate_fn=collate)

model = TransformerVA(input_dim=FEATURE_DIM)
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

def train():
    model.train()
    size = len(train_loader.dataset)

    for batch, (X, y) in enumerate(train_loader):
        pred_y = model(X)
        loss = F.mse_loss(pred_y, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * BATCH + len(X)
            print(f"loss: {loss} | {current}/{size}")

    torch.save(model.state_dict(), "../model_ckpts/transformer_va2.pt")
    print("Saved model.")

losses = []

def test():
    batches = len(test_data)
    datasize = len(test_data.dataset)
    test_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            pred_y = model(X)
            loss = F.mse_loss(pred_y, y)
            test_loss += loss.item()
    test_loss /= batches

    losses.append(test_loss)

    print(f"Testing Error: \n Average loss: {test_loss} \n")


if __name__ == "__main__":
    for i in range(EPOCHS):
        print(f"Epoch {i+1} \n -------------------------- \n")
        train()
        test()
    
    fig, lossplot = plt.subplot(), plt.subplot()
    lossplot.set_title("Average Loss and Accuracy over Epochs")
    lossplot.plot(range(EPOCHS), losses, color="orange", label="Loss")
    lossplot.set_xlabel("Epochs")
    lossplot.set_ylabel("Loss")
    lossplot.tick_params(axis="y", color="orange")
    lossplot.legend(loc="upper left")

    plt.savefig("resultsplot.png")


