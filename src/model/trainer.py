import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device=None, loss_fn=None):
        """
        Initializes the Trainer class.
        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            device (torch.device, optional): The device (CPU or GPU) to use for training.
            loss_fn (torch.nn.Module, optional): Loss function to use for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()  # Default loss function
    
        # Move model to the correct device
        self.model.to(self.device)

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, targets in self.train_loader:
            # Move data to the device (GPU/CPU)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.loss_fn(outputs, targets)
            running_loss += loss.item()

            # Backward pass (compute gradients)
            loss.backward()
            
            # Optimize the model parameters
            self.optimizer.step()
            
            # Track correct predictions
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            total_predictions += targets.size(0)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def validate_epoch(self):
        """
        Validate the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # No gradient tracking during validation
            for inputs, targets in self.val_loader:
                # Move data to the device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute the loss
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

                # Track correct predictions
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_predictions += targets.size(0)

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs, with validation at each epoch.
        Args:
            num_epochs (int): The number of epochs to train the model.
        """
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch()
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch()
            
            # Scheduler step (if provided)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save model checkpoint if the validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch, val_loss):
        """
        Save the model checkpoint.
        Args:
            epoch (int): The epoch at which the model is saved.
            val_loss (float): The validation loss at the time of saving.
        """
        checkpoint_path = f"checkpoint_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Model checkpoint loaded from {checkpoint_path}")

