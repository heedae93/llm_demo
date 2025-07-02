import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ 데이터 준비
corpus = ["나는 사과를 먹었고 딸기도 먹었지만 오렌지도 먹고 싶다"]
tokens = corpus[0].split()
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# 2️⃣ (입력, 정답) 쌍 만들기
data = []
for i in range(len(tokens) - 1):
    x = torch.tensor([[word2idx[tokens[i]]]])  # [batch=1, seq_len=1]
    y = torch.tensor([word2idx[tokens[i + 1]]])  # [batch=1]
    data.append((x, y))

# 학습 데이터 확인 출력
print("📦 학습 데이터셋 (입력 → 정답):")
for i, (x, y) in enumerate(data):
    print(f"{i+1}) {idx2word[x.item()]} → {idx2word[y.item()]}")

# 3️⃣ Transformer 언어모델 정의
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: [batch, seq_len]
        x = self.embed(x)            # [batch, seq_len, d_model]
        x = self.transformer(x)      # [batch, seq_len, d_model]
        out = self.fc(x[:, -1, :])   # 마지막 위치 출력만 사용 → [batch, vocab_size]
        return out

# 4️⃣ 모델 초기화
vocab_size = len(vocab)
model = TransformerLM(vocab_size, d_model=16, nhead=2, num_layers=1)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 총 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🔢 모델 총 파라미터 수: {total_params}")

# 5️⃣ 학습 전 가중치 일부 확인
print("\n🧊 학습 전 fc.weight 일부:")
print(model.fc.weight[:3])

# 6️⃣ 학습 루프
print("\n🔁 학습 시작")
for epoch in range(10):
    total_loss = 0
    print(f"\n🌀 Epoch {epoch + 1}")
    for i, (x, y) in enumerate(data):
        optimizer.zero_grad()
        pred = model(x)            # [1, vocab_size]
        loss = loss_fn(pred, y)    # [1, vocab_size] vs [1]
        loss.backward()

        # 가중치 변화 추적
        target_word_idx = 0
        target_word = idx2word[target_word_idx]
        weight_before = model.fc.weight.data.clone()[target_word_idx][:3]

        optimizer.step()
        weight_after = model.fc.weight.data.clone()[target_word_idx][:3]

        print(f"  📥 입력: {idx2word[x.item()]} ({x.item()})")
        print(f"  🎯 정답: {idx2word[y.item()]} ({y.item()})")
        print(f"  🧠 예측: {idx2word[pred.argmax().item()]} ({pred.argmax().item()})")
        print(f"  ❌ 손실: {loss.item():.6f}")
        print(f"  🧊 fc.weight['{target_word}'] (전): {weight_before}")
        print(f"  🔥 fc.weight['{target_word}'] (후): {weight_after}\n")

        total_loss += loss.item()

    print(f"✅ Epoch {epoch + 1} 종료, 평균 손실: {total_loss / len(data):.4f}")

# 7️⃣ 학습 후 가중치 일부 출력
print("\n✅ 학습 완료! fc.weight 일부:")
print(model.fc.weight[:3])

# 8️⃣ 예측 함수
def predict(word):
    if word not in word2idx:
        print(f"❌ 단어 '{word}'는 단어장에 없습니다.")
        return
    with torch.no_grad():
        x = torch.tensor([[word2idx[word]]])  # [batch=1, seq_len=1]
        out = model(x)
        pred_idx = torch.argmax(out, dim=1).item()
        print(f"🔮 예측: '{word}' → '{idx2word[pred_idx]}'")

# 9️⃣ 예측 실행
print("\n🔍 예측 테스트:")
predict("나는")
predict("사과를")
predict("먹었고")
predict("딸기도")
predict("먹었지만")
predict("오렌지도")
predict("먹고")
predict("싶다")