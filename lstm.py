import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

config = {
    "BATCH_SIZE": 4,
    "MAX_LEN": 128,
    "VOCAB_SIZE": tokenizer.vocab_size,
    "EMBEDDING_DIM": 512,
    "HIDDEN_SIZE": 512,
    "NUM_CLASSES": 3,
    "LEARNING_RATE": 1e-3,
    "NUM_EPOCHS": 8,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        output_size,
        bidirectional,
        average_hidden,
    ):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

        if bidirectional:
            hidden_size *= 2  # If bidirectional, double the hidden size

        self.avg_hidden = average_hidden
        self.fc = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc2 = nn.Linear(int(hidden_size / 2), output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        if self.avg_hidden:
            lstm_out = lstm_out.mean(dim=1)

        output = self.fc(lstm_out)
        output = self.relu(output)
        output = self.fc2(output)
        return output


# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]["sentence"]
        label = self.data.iloc[idx]["label"]
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Function to train the model
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = (
            batch["input_ids"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Function to evaluate the model
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = (
                batch["input_ids"].to(device),
                batch["label"].to(device),
            )
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()
    accuracy = correct_predictions / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy


# Load data from CSV files
train_data = pd.read_csv("train_name.csv")
dev_data = pd.read_csv("dev_name.csv")
test_data = pd.read_csv("test_name.csv")

# Split data into features and labels
X_train, y_train = train_data["sentence"], train_data["label"]
X_dev, y_dev = dev_data["sentence"], dev_data["label"]
X_test, y_test = test_data["sentence"], test_data["label"]

print("Data max length: ", max([len(x.split()) for x in X_train]))

# Tokenize data and create DataLoader
train_dataset = SentimentDataset(train_data, tokenizer, max_len=config["MAX_LEN"])
dev_dataset = SentimentDataset(dev_data, tokenizer, max_len=config["MAX_LEN"])
test_dataset = SentimentDataset(test_data, tokenizer, max_len=config["MAX_LEN"])

train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

# Initialize and train the model
model = SentimentLSTM(
    vocab_size=config["VOCAB_SIZE"],
    embedding_dim=config["EMBEDDING_DIM"],
    hidden_size=config["HIDDEN_SIZE"],
    output_size=config["NUM_CLASSES"],
    bidirectional=True,
    average_hidden=True,
).to(config["DEVICE"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

best_dev_accuracy = 0.0

train_losses = []
dev_losses = []

for epoch in range(config["NUM_EPOCHS"]):
    train_loss = (train(model, train_loader, criterion, optimizer, config["DEVICE"]))
    dev_loss, dev_accuracy = (
        evaluate(model, test_loader, criterion, config["DEVICE"]),
    )
    print(
        f"Epoch {epoch + 1}/{config['NUM_EPOCHS']}: "
        f"Train Loss: {train_loss:.4f}, "
        f"Dev Loss: {dev_loss:.4f}, "
        f"Dev Accuracy: {dev_accuracy:.2%}"
    )
    train_losses.append(train_loss)
    dev_losses.append(dev_loss)

    # Save the best model based on development set accuracy
    if dev_accuracy > best_dev_accuracy:
        best_dev_accuracy = dev_accuracy
        torch.save(model.state_dict(), "best_model.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")

# Load the best model for evaluation on the test set
best_model = SentimentLSTM(
    vocab_size=config["VOCAB_SIZE"],
    embedding_dim=config["EMBEDDING_DIM"],
    hidden_size=config["HIDDEN_SIZE"],
    output_size=config["NUM_CLASSES"],
    bidirectional=True,
    average_hidden=True,
).to(config["DEVICE"])
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.eval()

# Evaluate on the test set
test_loss, test_accuracy = evaluate(
    best_model, test_loader, criterion, config["DEVICE"]
)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")

# Print the classification report
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Generating predictions", leave=False):
        inputs, labels = batch["input_ids"].to(config["DEVICE"]), batch["label"].to(
            config["DEVICE"]
        )
        outputs = best_model(inputs)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(
    classification_report(
        all_labels, all_predictions, target_names=["0", "1", "2"], digits=4
    )
)
