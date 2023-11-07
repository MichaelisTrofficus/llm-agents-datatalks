import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import generate_ratings, RatingsDataset
from model import MatrixFactorization

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate ratings
ratings_df = generate_ratings()

# Create dataset and dataloader
dataset = RatingsDataset(ratings_df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create model
model = MatrixFactorization(1000, 1000).to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in tqdm(range(100)):
    for user, item, rating in dataloader:
        # Move data to device
        user = user.to(device)
        item = item.to(device)
        rating = rating.float().to(device)

        # Forward pass
        outputs = model(user, item)
        loss = criterion(outputs, rating)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), './model.pth')

print('Training complete. Model saved to ./model.pth')
