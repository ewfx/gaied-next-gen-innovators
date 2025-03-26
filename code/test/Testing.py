

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
