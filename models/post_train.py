import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def step(self, model, input, target, config):
        # Get the model's output
        output = model(input)
        
        # Compute the reward
        reward = reward_function(output, target, config)
        
        # Return the output and reward
        return output, reward

# Step 5: Train the Model
def train(model, env, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for input, target in env.dataset:
            # Forward pass
            output = model(input)
            
            # Compute the reward
            reward = reward_function(output, target, config)
            
            # Compute the loss (negative log probability weighted by reward)
            log_probs = torch.log_softmax(output, dim=1)
            selected_log_probs = log_probs.gather(1, target.unsqueeze(1))
            loss = -selected_log_probs.mean() * reward
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for logging
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {epoch_loss / len(env.dataset):.4f}")

# Step 6: Evaluate the Model
def evaluate(model, env, config):
    print("\nEvaluating the model:")
    with torch.no_grad():
        for input, target in env.dataset:
            output = model(input)
            predicted = torch.argmax(output, dim=1)
            reward = reward_function(output, target, config)
            print(f"Input: {input.shape}, Target: {target.item()}, Predicted: {predicted.item()}, Reward: {reward.item()}")

# Step 7: Save the Model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"\nFine-tuned model saved to '{filename}'")

# Step 8: Load and Use the Model for Inference
def inference(model, config):
    print("\nUsing the fine-tuned model for inference:")
    example_input = torch.randn(config.batch_size, config.input_size)
    with torch.no_grad():
        output = model(example_input)
        predicted = torch.argmax(output, dim=1)
        print(f"Example Input: {example_input.shape}, Predicted Class: {predicted.item()}")

# Main Function
def main():
    # Initialize config, model, and environment
    config = Config()
    model = CustomModel(config)
    env = Environment(config)
    
    # Train the model
    train(model, env, config)
    
    # Evaluate the model
    evaluate(model, env, config)
    
    # Save the fine-tuned model
    save_model(model, "fine_tuned_rl_model.pth")
    
    # Load and use the model for inference
    model = CustomModel(config)
    model.load_state_dict(torch.load("fine_tuned_rl_model.pth"))
    model.eval()
    inference(model, config)

if __name__ == "__main__":
    main()
