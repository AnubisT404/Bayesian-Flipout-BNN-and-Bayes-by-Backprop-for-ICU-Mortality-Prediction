import torch
from torch import nn
import torch.nn.functional as F
import bayesian_torch.layers as bnn_layers
import argparse
from tqdm import tqdm
from train_utils import load_datasets

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlipoutBNN(nn.Module):
    def __init__(self, in_dim, n_classes, prior_sigma=0.1):
        super().__init__()
        self.l1 = bnn_layers.LinearFlipout(
            in_dim, 128, prior_mu=0.0, prior_sigma=prior_sigma
        )
        self.l2 = bnn_layers.LinearFlipout(
            128, 128, prior_mu=0.0, prior_sigma=prior_sigma
        )
        self.l3 = bnn_layers.LinearFlipout(
            128, n_classes, prior_mu=0.0, prior_sigma=prior_sigma
        )
        self.nll = nn.BCEWithLogitsLoss(reduction="sum")  # Binary classification loss

    def forward(self, x):
        x = x.view(len(x), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


def flipout_elbo_loss(model, x, y, kl_weight=1.0):
    """Compute ELBO loss for Bayesian NN with Flipout."""
    output = model(x)
    nll = model.nll(output.squeeze(-1), y)  # Negative log likelihood (BCE)

    # Fix: Compute KL loss correctly
    kl = sum(
        layer.kl_loss()[0] + layer.kl_loss()[1]
        for layer in [model.l1, model.l2, model.l3]
    )

    return nll + kl_weight * kl, nll, kl


def train_epoch(model, loader, optimizer, kl_weight=1.0):
    """Train for one epoch using ELBO loss."""
    model.train()
    total_loss = 0
    for data_x, data_y in tqdm(loader, desc="Training"):
        data_x, data_y = data_x.to(DEVICE), data_y.to(DEVICE)
        optimizer.zero_grad()
        elbo, nll, kl = flipout_elbo_loss(model, data_x, data_y, kl_weight=kl_weight)
        elbo.backward()
        optimizer.step()
        total_loss += elbo.item()
    return total_loss / len(loader)


def validate(model, loader, samples=20, sample_batch_size=64):
    """Evaluate model uncertainty over multiple forward passes."""
    model.eval()
    num_samples = len(loader.dataset)

    # Fix: Pre-allocate output tensor correctly
    outputs = torch.zeros(samples, num_samples, model.n_classes).to(DEVICE)

    with torch.no_grad():
        for i in range(samples):
            all_outputs = []
            for data_x, _ in tqdm(loader, desc=f"Sampling {i + 1}/{samples}"):
                data_x = data_x.to(DEVICE)
                batch_outputs = model(data_x)
                all_outputs.append(batch_outputs)

            outputs[i] = torch.cat(all_outputs, dim=0)  # Fix for batch size mismatch

    predictions = torch.sigmoid(outputs.mean(dim=0)).squeeze(-1).cpu().numpy()
    uncertainties = outputs.std(dim=0).squeeze(-1).cpu().numpy()

    return predictions, uncertainties


def main(
    data_path: str,
    model_path: str,
    train_batch_size: int,
    test_batch_size: int,
    lr: float,
    epochs: int,
):
    """Main function for training Bayesian Flipout model."""
    # Load dataset
    train_dataset, test_dataset = load_datasets(data_dir=data_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    # Model initialization
    model = FlipoutBNN(in_dim=44, n_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model uncertainty
    predictions, uncertainties = validate(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="../data/processed", help="Path to dataset"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../models/flipout_bnn.pth",
        help="Model save path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=512, help="Test batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=256, help="Number of epochs")
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_path,
        args.batch_size,
        args.test_batch_size,
        args.lr,
        args.epochs,
    )
