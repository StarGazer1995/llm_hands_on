import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import importlib.util
import sys
import os
import inspect

class Trainer:
    def __init__(self, arg):
        cfg_path = arg.config
        self.tag = arg.tag
        self.config_path = cfg_path
        self.cfg = self.load_config(cfg_path)
        self.model_cfg = self.cfg
        self.update_config()

        self.build_model()
        self.build_optimizer()
        self.env = Environment(self.model_cfg)

    def update_config(self):
        self.batch_size = self.model_cfg.batch_size
        self.num_epochs = self.model_cfg.num_epochs
        self.learning_rate = self.model_cfg.learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_model(self):
        self.model = CustomModel(self.model_cfg).to(self.device)

    def load_config(self, config_file_path):
        # Load the configuration file as a module
        spec = importlib.util.spec_from_file_location("config_module", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)
        # Return the module for accessing the configuration
        return config_module

    def save_model(self, epoch):
        workspace_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        base_file_name = self.config_path.split("/")[-1]
        filename = base_file_name.split(".")[0]
        save_dir = os.path.join(workspace_dir, "ckpts", filename)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except:
            raise FileNotFoundError
        
        if self.tag is not None:
            filename = f"epoch_{str(epoch)}_{self.tag}"
        else:
            filename = f"epoch_{str(epoch)}"
        source = inspect.getsource(self.cfg)
        filename = os.path.join(save_dir, filename)

        torch.save(self.model.state_dict(), filename + ".pth")
        with open(f"{filename}.py", "w") as f:
            f.write(source)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for input, target in self.env.dataset:
                input, target = input.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(input)
                
                # Compute the reward
                reward = reward_function(output, target, self.model_cfg)
                
                # Compute the loss (negative log probability weighted by reward)
                log_probs = torch.log_softmax(output, dim=1)
                selected_log_probs = log_probs.gather(1, target.unsqueeze(1))
                loss = -selected_log_probs.mean() * reward
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss for logging
                epoch_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {epoch_loss / len(self.env.dataset):.4f}")

        self.save_model("last")

    def evaluate(self):
        print("\nEvaluating the model:")
        self.model.eval()
        with torch.no_grad():
            for input, target in self.env.dataset:
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                predicted = torch.argmax(output, dim=1)
                reward = reward_function(output, target, self.model_cfg)
                print(f"Input: {input.shape}, Target: {target.item()}, Predicted: {predicted.item()}, Reward: {reward.item()}")
        self.model.train()

    def inference(self):
        print("\nUsing the fine-tuned model for inference:")
        self.model.eval()
        example_input = torch.randn(self.batch_size, self.model_cfg.input_size).to(self.device)
        with torch.no_grad():
            output = self.model(example_input)
            predicted = torch.argmax(output, dim=1)
            print(f"Example Input: {example_input.shape}, Predicted Class: {predicted.item()}")
        self.model.train()

# Step 1: Define the Config Class
class Config:
    # Model hyperparameters
    input_size = 768  # Example input size (e.g., BERT embeddings)
    hidden_size = 256
    output_size = 2   # Binary output (e.g., positive/negative sentiment)
    
    # Training hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 1  # Since the dataset is small
    
    # Reward function hyperparameters
    reward_correct = 1.0   # Reward for correct prediction
    reward_incorrect = -1.0  # Penalty for incorrect prediction
    
    # Environment hyperparameters
    dataset_size = 2  # Number of (input, target) pairs in the dataset

# Step 2: Define the Custom Model
class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 3: Define the Reward Function
def reward_function(output, target, config):
    # Reward +1 for correct prediction, -1 for incorrect
    predicted = torch.argmax(output, dim=1)
    reward = torch.where(predicted == target, torch.tensor(config.reward_correct), torch.tensor(config.reward_incorrect))
    return reward

# Step 4: Define the Environment
class Environment:
    def __init__(self, config):
        # Generate a synthetic dataset of (input, target) pairs
        self.dataset = [
            (torch.randn(config.batch_size, config.input_size), torch.randint(0, config.output_size, (config.batch_size,)))
            for _ in range(config.dataset_size)
        ]

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--tag")
    return parser.parse_args()

if __name__ == "__main__":
    arg = parseargs()
    trainer = Trainer(arg)
    trainer.train()
    trainer.evaluate()
    trainer.inference()
