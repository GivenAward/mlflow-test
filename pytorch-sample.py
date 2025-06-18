import os

import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

load_dotenv()  # Load environment settings
mlflow.set_tracking_uri("http://192.168.0.100:5000")  # Set your MLflow tracking URI
mlflow.set_experiment("pytorch-mnist-example")  # Set your MLflow experiment name


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            acc = 100.0 * correct / total if total > 0 else 0.0
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.3f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    acc,
                )
            )
            if args.dry_run:
                break
    mlflow.log_metric("train_loss", loss.item(), step=epoch)
    mlflow.log_metric("train_acc", acc, step=epoch)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    mlflow.log_metrics(
        {
            "valid_loss": test_loss,
            "val_acc": 100.0 * correct / len(test_loader.dataset),
        }
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 8, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    mlflow.pytorch.autolog()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "batch_size": args.batch_size,
                "test_batch_size": args.test_batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "gamma": args.gamma,
                "seed": args.seed,
                "log_interval": args.log_interval,
                "save_model": args.save_model,
            }
        )
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
        if args.save_model:
            test_batch = next(iter(test_loader))
            test_x, _ = test_batch
            test_x = test_x.to(device)
            test_pred = model(test_x)
            signature = infer_signature(test_x.cpu().numpy(), test_pred.cpu().detach().numpy())
            torch.save(model.state_dict(), "mnist_cnn.pt")
            mlflow.pytorch.log_model(
                model, "model", signature=signature, pip_requirements=["torch==2.5.1", "cloudpickle==2.2.1"]
            )
            print("Model saved as mnist_cnn.pt")

        model.eval()
        with torch.no_grad():
            test_datapoint, test_target = next(iter(test_loader))
            prediction = model(test_datapoint.to(device))
            actual = test_target[0].item()
            predicted = torch.argmax(prediction).item()
            print(f"\nPREDICTION RESULT: ACTUAL: {actual!s}, PREDICTED: {predicted!s}")


if __name__ == "__main__":
    main()
