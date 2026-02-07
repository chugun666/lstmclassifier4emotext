import torch
import torch.nn as nn
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.15)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out)

def load_assets(folder='saved_model_1'):
    with open(f'{folder}/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open(f'{folder}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    num_classes = len(le.classes_)
    model = LSTMClassifier(len(vocab), 128, 256, num_classes).to(device)
    
    model.load_state_dict(torch.load(f'{folder}/lstm_weights.pth', map_location=device))
    model.eval() 
    
    return model, vocab, le

def predict(text, model, vocab, le):
    tokens = str(text).lower().split()
    indices = [vocab.get(w, 1) for w in tokens] 
    
    input_tensor = torch.tensor([indices]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][idx].item()
    
    label = le.inverse_transform([idx])[0]
    return label, confidence

model, vocab, le = load_assets()

test_comments = [
    "ENTER YOUR TEXT",
]

print(f"{'Comment':<50} | {'Prediction':<12} | {'Confidence'}")
print("-" * 80)

for comment in test_comments:
    category, conf = predict(comment, model, vocab, le)
    short_text = (comment[:47] + '..') if len(comment) > 47 else comment
    print(f"{short_text:<50} | {category:<12} | {conf:.2%}")