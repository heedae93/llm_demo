import torch
import torch.nn as nn
import torch.optim as optim

# 1️⃣ 학습 데이터 정의
corpus = ["나는 나는 사과를 먹었다"]

# 2️⃣ 단어 사전 생성 , 컴퓨터가 이해할 수 있도록 텍스트를 숫자로 변환해야 한다.
token_list = list(set(" ".join(corpus).split())) # 공백 기준으로 단어 분리 하고 중복 단어 제거
token_list.sort() # 사전순 정렬
word2idx = {w: i for i, w in enumerate(token_list)}
idx2word = {i: w for w, i in word2idx.items()} # 단어를 key로 하고 값이 index인 딕셔너리 생성
print("단어 인덱스 매핑:", word2idx)

# 3️⃣ 훈련 데이터 생성 (입력 단어 → 다음 단어 예측) , 입력과 정답 쌍 ( x, y ) 를 만드는 과정
data = []
tokens = corpus[0].split()
for i in range(len(tokens) - 1):
    x = torch.tensor([word2idx[tokens[i]]])    # 입력 단어
    y = torch.tensor([word2idx[tokens[i + 1]]])  # 정답 단어
    data.append((x, y))
print("\n학습 샘플:")



for x, y in data:
    print(f"입력: {idx2word[x.item()]} → 정답: {idx2word[y.item()]}")



# 4️⃣ 아주 간단한 LSTM 언어모델 정의
class MiniLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # 단어를 임베딩 벡터로 변환
        x, _ = self.lstm(x.view(len(x), 1, -1))  # LSTM 처리
        out = self.fc(x[-1])  # 마지막 시간의 출력 → 분류
        return out

# 5️⃣ 모델, 손실함수, 옵티마이저 설정
vocab_size = len(word2idx)
model = MiniLSTM(vocab_size, embed_dim=10, hidden_dim=20)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6️⃣ 학습 전 초기 가중치 일부 출력
print("\n✅ 초기 가중치 샘플 (fc.weight):")
print(model.fc.weight[:3])  # 일부 출력

# 7️⃣ 학습 루프
print("\n🔁 학습 시작")
for epoch in range(10):
    total_loss = 0
    for x, y in data:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()     # 🔧 역전파: 손실 기반 파라미터 변화량 계산
        optimizer.step()    # 💡 파라미터 갱신 (가중치 업데이트)
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 8️⃣ 학습 후 가중치 출력
print("\n✅ 학습 후 가중치 샘플 (fc.weight):")
print(model.fc.weight[:3])  # 일부 출력

# 9️⃣ 예측 확인
def predict(word):
    if word not in word2idx:
        print(f"단어 '{word}'는 사전에 없습니다.")
        return
    with torch.no_grad():
        x = torch.tensor([word2idx[word]])
        out = model(x)
        predicted_idx = torch.argmax(out).item()
        print(f"{word} → {idx2word[predicted_idx]}")

print("\n🔮 예측 결과:")
predict("나는")
predict("사과를")
