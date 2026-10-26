import mlflow
import torch

# Use SQLite backend instead of file store for MLflow 3.x compatibility
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("dino-ml-local-test")

with mlflow.start_run():
    # Define simple training parameters
    learning_rate = 0.01
    epochs = 5
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)

    # Create a dummy PyTorch tensor on CPU
    x = torch.randn(100, 10)
    y = torch.matmul(x, torch.randn(10, 1))

    # Simulate training and log metrics
    for epoch in range(epochs):
        loss = 0.5 / (epoch + 1)
        mlflow.log_metric("loss", loss, step=epoch)

    print("MLflow local test completed successfully!")
