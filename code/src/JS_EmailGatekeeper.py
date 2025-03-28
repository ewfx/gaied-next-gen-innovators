import os
import json
import torch
import spacy
import pdfplumber
import email
from email import policy
from email.parser import BytesParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Paths
EML_FOLDER = "C:/Users/motat/Desktop/JS_EmailGatekeeper/JS_EmailData/JS_Emails"
JSON_PATH = "C:/Users/motat/Desktop/JS_EmailGatekeeper/JS_EmailData/requests.json"

# Load dataset
with open(JSON_PATH, 'r') as f:
    label_data = json.load(f)

def extract_text_from_eml(eml_path):
    """Extract plain text from .eml files."""
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return msg.get_body(preferencelist=('plain', 'html')).get_content() if msg.get_body() else ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF attachments."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_key_attributes(text):
    """Extract key information like loan numbers, dates, and amounts."""
    doc = nlp(text)
    loan_numbers, dates, amounts = [], [], []
    
    for ent in doc.ents:
        if ent.label_ == "CARDINAL" and len(ent.text) >= 6:
            loan_numbers.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "MONEY":
            amounts.append(ent.text)
    
    return {
        "loan_account_number": loan_numbers,
        "dates": dates,
        "amounts": amounts
    }
    

# Process dataset
texts, labels = [], []
all_labels = set()

for filename, data in label_data.items():
    if "requests" in data:
        for request in data["requests"]:
            all_labels.add(request.get("sub_request_type", "unknown"))
    else:
        all_labels.add(data.get("sub_request_type", "unknown"))

mlb = MultiLabelBinarizer()
mlb.fit([[label] for label in all_labels])

for filename, data in label_data.items():
    eml_path = os.path.join(EML_FOLDER, filename)
    if os.path.exists(eml_path):
        email_text = extract_text_from_eml(eml_path)
        texts.append(email_text)
        
        if "requests" in data:
            label_list = [request.get("sub_request_type", "unknown") for request in data["requests"]]
        else:
            label_list = [data.get("sub_request_type", "unknown")]
        
        labels.append(mlb.transform([label_list])[0])

labels = torch.tensor(labels, dtype=torch.float)

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(mlb.classes_), problem_type="multi_label_classification")

# Dataset class
class EMLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = EMLDataset(train_texts, train_labels, tokenizer)
test_dataset = EMLDataset(test_texts, test_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete. Model saved.")



#Testing
def predict(text):
    """Run inference on an email and extract structured data."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_index = torch.argmax(logits, dim=-1).item()  # Get highest probability index
    sub_request_type = mlb.classes_[predicted_index]

    # Extract key attributes
    key_attributes = extract_key_attributes(text)

    # Find corresponding request_type
    request_type = None
    for filename, data in label_data.items():
        if "requests" in data:
            for request in data["requests"]:
                if request["sub_request_type"] == sub_request_type:
                    request_type = request["request_type"]
                    break
        else:
            if data["sub_request_type"] == sub_request_type:
                request_type = data["request_type"]

    return {
        "request_type": request_type,
        "sub_request_type": sub_request_type,
        "key_attributes": key_attributes
    }

# Example Test
test_email_text = "I need a loan modification to lower my monthly payments due to financial hardship."
predicted_result = predict(test_email_text)
print("Predicted Output:", json.dumps(predicted_result, indent=4))
#If we want to test for group of emails
for filename in os.listdir(EML_FOLDER):
    if filename.endswith(".eml"):
        email_text = extract_text_from_eml(os.path.join(EML_FOLDER, filename))
        predictions = predict(email_text)
        print(f"Email: {filename} -> Predicted Categories: {predictions}")

