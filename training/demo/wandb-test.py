# See: https://wandb.ai/wandb_fc/articles/reports/Multi-GPU-Hyperparameter-Sweeps-in-Three-Simple-Steps--Vmlldzo1NDQ0MDEy
import wandb

# Initialize a new W&B run
wandb.init()

# Access the parameters from the YAML file via W&B
params = wandb.config


def run_model(params):
    results = []
    for epoch in range(1, 4):  # Running for 3 epochs
        # Does not include auc
        loss = (
            params["audc"]
            + params["class_connection"]
            + params["stability"]
            + params["semantic_similarity"]
            + epoch
        )
        results.append(loss)
        print(f"Epoch {epoch}, Loss: {loss}")
    return results


# Perform the run
losses = run_model(params)

# Log the losses to W&B
for epoch, loss in enumerate(losses, start=1):
    wandb.log({"epoch": epoch, "loss": loss})

wandb.finish()
