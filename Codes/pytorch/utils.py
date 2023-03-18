import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


def load_data(batch_size=32):
    to_tensor = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=to_tensor
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=to_tensor
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def train_step(model, loss_fn, optimizer, x, y):
    model.train()
    yhat = model(x)
    l = loss_fn(yhat, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    return l.item()


def test_step(model, loss_fn, x, y):
    model.eval()
    with torch.inference_mode():
        yhat = model(x)
        l = loss_fn(yhat, y)
    return l.item()


def accuracy(model, x, y):
    model.eval()
    yhat = model(x)
    yhat = torch.argmax(yhat, dim=1)
    return torch.sum(yhat == y).item() / len(y)


def predict(model, x):
    model.eval()
    yhat = model(x)
    yhat = torch.argmax(yhat, dim=1)
    return yhat


def batch_progress(cur_batch, all_batch, loss):
    """Prints an arrow showing progress about the current batch.

    Parameters
    ----------
    cur_batch : int
        Current batch number.
    all_batch : int
        Total number of batches.
    loss : float
        Current loss. Should be averaged over all batches.
    epoch : int
        Current epoch number.
    all_epoch : int
        Total number of epochs.

    Returns
    -------
    String
        The progress bar. In a form of 10/100[====----]
    """
    len_progress_bar = 20
    progress = int((cur_batch + 1) / all_batch * len_progress_bar)
    progress_bar = "=" * progress + "-" * (len_progress_bar - progress)
    digits = len(str(all_batch))
    return f"Batch: {cur_batch:>{digits}}/{all_batch:>{digits}} | [{progress_bar}] | Loss: {loss:.4f}"


def epoch_progress(epoch, all_epoch, metrics):
    """Prints an arrow showing progress about the current epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    all_epoch : int
        Total number of epochs.
    metrics : dict
        Dictionary containing the metrics.

    Returns
    -------
    String
        The progress bar. In a form of 10/100[===>----]
    """
    len_progress_bar = 20
    progress = int((epoch + 1) / all_epoch * len_progress_bar)
    progress_bar = "=" * progress + "-" * (len_progress_bar - progress)
    digits = len(str(all_epoch))
    cur_text = f"Epoch: {epoch+1:>{digits}}/{all_epoch:>{digits}} | [{progress_bar}] | "

    for key, value in metrics.items():
        cur_text += f"{key}: {value:.4f} | "

    return cur_text[:-3]
