import argparse
import time
from model import *
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs, model, loss_fn, optimizer, train_loader, test_loader):
    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            train_loss += train_step(model, loss_fn, optimizer, x, y)
            b_progress = batch_progress(
                i,
                len(train_loader),
                train_loss / (i + 1),
                metrics=None,
            )
            print(b_progress, end="\r")

        train_acc = accuracy(model, x, y)
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            test_loss += test_step(model, loss_fn, x, y)
        test_acc = accuracy(model, x, y)
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        e_progress = epoch_progress(epoch, epochs, metrics)
        print(e_progress)


def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--model", type=str, default="model0", help="Model name, one of: model0, conv1"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=64, help="Number of hidden units"
    )
    parser.add_argument(
        "--color_channel", type=int, default=1, help="Number of color channels"
    )
    parser.add_argument(
        "--output_shape", type=int, default=10, help="Number of output classes"
    )
    return parser


def load_args():
    parser = arg_parser()
    args = parser.parse_args()
    return args


def main():
    print(f"Using device: {DEVICE}")
    args = load_args()
    hidden_units = args.hidden_units
    color_channel = args.color_channel
    output_shape = args.output_shape
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    if args.model == "model0":
        model = Model0()
    elif args.model == "conv1":
        model = Conv1(
            hidden_units=hidden_units,
            color_channel=color_channel,
            output_shape=output_shape,
        )
    else:
        raise ValueError("Invalid model name")

    # print the model
    model = model.to(DEVICE)
    print(model)

    # Load data
    train_loader, test_loader = load_data(batch_size=batch_size)

    # Define loss function
    loss = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    train(epochs, model, loss, optimizer, train_loader, test_loader)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print(f"Time elapsed: {toc - tic:.2f} seconds")
