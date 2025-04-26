import torch
import torch.nn as nn
import torch.optim as optim
from model import SmileNet
from dataset import SmileDataset, unicornLoader  
from config import epochs, learning_rate, model_save_path
import os


print("Starting training...")

def my_descriptively_named_train_function():
    try:
        
        print("Initializing model and optimizer...")  
        model = SmileNet()
        criterion = nn.BCELoss()  
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
       
        print("Loading data loader...")  
        train_loader = unicornLoader()  
        
        
        print(f"Total batches in train_loader: {len(train_loader)}")
        
        
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"First batch - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            break  

        model.train()  
        print("Starting training loop...")  
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}...")  

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                print(f"Batch {i+1} of {len(train_loader)}")  

                
                outputs = model(inputs)
                
                
                loss = criterion(outputs, labels.float().unsqueeze(1))

                
                optimizer.zero_grad()

                
                loss.backward()
                optimizer.step()

                
                running_loss += loss.item()

            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

        
        print("Saving model...")  
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        
        print(f"Saving model to: {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        print(f"\nModel saved to {model_save_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    my_descriptively_named_train_function()
