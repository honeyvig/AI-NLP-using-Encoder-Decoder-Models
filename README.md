# AI-NLP-using-Encoder-Decoder-Models
To work with a dataset from Hugging Face and train a model with a classification layer using an encoder-only model, followed by a decoder model, we will follow the steps below. For this example, we will use transformers from Hugging Face and PyTorch to implement the required process.

Here are the steps to:

    Load the dataset.
    Train and test a model (typically, this would be a pre-trained model like BERT or RoBERTa).
    Add a classification layer after the encoder.
    Add a decoder for sequence generation or other tasks.

Step 1: Install Required Libraries

First, you need to install the necessary Python packages:

pip install transformers datasets torch scikit-learn

Step 2: Load the Dataset

We'll use Hugging Face's datasets library to load the dataset. Let’s assume the dataset is a classification dataset, but the code can be easily modified for other tasks.

from datasets import load_dataset

# Load the dataset (replace 'your_dataset_name' with the actual dataset name)
dataset = load_dataset('your_dataset_name')

# Check the dataset structure (train/test split)
print(dataset)

Step 3: Preprocess the Data

Now, we will preprocess the data. We will use a pre-trained encoder model (e.g., BERT) from Hugging Face, tokenize the dataset, and prepare it for training.

from transformers import BertTokenizer

# Load the tokenizer for BERT (or another encoder-only model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Apply tokenization to the train and test splits
train_dataset = dataset['train'].map(tokenize_function, batched=True)
test_dataset = dataset['test'].map(tokenize_function, batched=True)

# Format the dataset for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

Step 4: Build the Encoder Model with Classification Layer

Now we will load a pre-trained encoder (like BERT) and add a classification layer on top. For this task, we will use BERT as an encoder and add a classification head.

from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Load the pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

Step 5: Training the Model

Now, we will train the model on the dataset.

# Training loop
epochs = 3  # You can adjust the number of epochs
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in train_dataloader:
        # Move the batch to the GPU if available
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        
        # Track accuracy
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += (preds == batch['label']).sum().item()
        total_predictions += len(batch['label'])

    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

Step 6: Testing and Evaluating the Model

Once the model is trained, we will test it on the validation/test dataset.

model.eval()  # Set the model to evaluation mode

correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        logits = outputs.logits
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += (preds == batch['label']).sum().item()
        total_predictions += len(batch['label'])

accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy}")

Step 7: Adding a Decoder Model

For a more sophisticated architecture, you can add a decoder layer to the model. Typically, this is done for sequence-to-sequence tasks (like translation or summarization). However, in the case of classification, this might not be necessary unless you're doing a sequence generation task.

If you still want to add a decoder model, you could extend the architecture with a BART or T5 model, both of which are encoder-decoder models.

Here’s how you would modify it for a sequence generation task (using an encoder-decoder model like BART or T5):

from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize BART or T5 for encoder-decoder tasks
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Example of preparing the input for a sequence-to-sequence task
input_text = "Translate English to French: How are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the output (e.g., translation)
outputs = model.generate(inputs['input_ids'])
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)

This approach will work for sequence-to-sequence tasks like translation, text generation, or summarization. If you need a classification model, you wouldn't typically use a decoder layer, but if the task requires outputting sequences (e.g., a text generation task after classification), adding the decoder makes sense.
Step 8: Answering Questions Regarding the Dataset

To answer questions regarding the dataset, you could analyze it with basic exploratory data analysis (EDA) techniques, such as:

# Checking the first few samples
print(dataset['train'][0])

# Check the distribution of labels (if it's a classification dataset)
import matplotlib.pyplot as plt

labels = dataset['train']['label']
plt.hist(labels, bins=len(set(labels)))
plt.show()

Conclusion

This approach loads a dataset from Hugging Face, tokenizes it, trains a model using a pre-trained encoder (BERT), and adds a classification layer. After training and testing the model, you can optionally integrate a decoder layer if your task requires sequence generation. This flexible pipeline can be adapted for different types of tasks such as classification or sequence-to-sequence generation.
