import torch
import torch.nn as nn
import torch.optim as optim


# 학습이란 딥러닝 모델이 주어진 텍스트 데이터의 순서를 통계적으로 기억하고 일반화 하는 과정



# 1️⃣ 학습 데이터 정의
corpus = ["나는 사과를 먹었다"]

# 2️⃣ 단어 사전 생성 ( 텍스트를 숫자로 변환해야 한다. 왜냐하면 다음 단계인 임베딩 과정에서 숫자 인덱스만 입력으로 받을 수 있기 때문 )
token_list = list(set(" ".join(corpus).split()))
token_list.sort()
# 각 단어에 인덱스 지정
word2idx = {w: i for i, w in enumerate(token_list)}
#  # 단어를 key로 하고 값이 index인 딕셔너리 생성 ( 이걸 하는 이유는 모델의 예측 결과가 숫자 인덱스로 나오기 때문에 그걸 다시 단어로 돌리기 위함 )
idx2word = {i: w for w, i in word2idx.items()}
# print("단어 인덱스 매핑:", word2idx)

# 3️⃣ 훈련 데이터 생성 ( 입력 단어 -> 정답 단어 쌍들의 집합 ) , 여기서 입력 단어는 이전 글자 정답 단어는 다음에 오는 글자
# tensor는 PyTorch에서 사용하는 기본 데이터 구조로, 쉽게 말해 **숫자를 담는 다차원 배열 예를들어 tensor([2]) 는 tensor 자료구조에 2라는 값이 들어가 있는 것.
data = []
tokens = corpus[0].split()
for i in range(len(tokens) - 1):
    # 입력 단어
    x = torch.tensor([word2idx[tokens[i]]])
    # 정답 단어
    y = torch.tensor([word2idx[tokens[i + 1]]])
    data.append((x, y))



print("📦 학습 전 최종 데이터 셋:")
for i, (x, y) in enumerate(data):
    print(f"{i+1}) 입력 텐서: {x}, 정답 텐서: {y}")

print("📦 data에 저장된 인덱스의 실제 문자열 값들 :")
for x, y in data:
    print(f"입력: {idx2word[x.item()]} → 정답: {idx2word[y.item()]}")



# 4️⃣ 아주 간단한 LSTM 언어모델 정의 ( PyTorch의 모델 기본형인 nn.Module을 상속받은 신경망 클래스 ) 및 세팅 , LSTM은 언어 모델을 만들기 위한 아키텍쳐
class MiniLSTM(nn.Module):

    # 생성자, 멤버 변수의 각 계층을 할당
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim) # 단어를 숫자 벡터로 바꿔주는 전처리기 (입력 계층)
        self.lstm = nn.LSTM(embed_dim, hidden_dim) # 단어 사이의 문맥을 이해하는 뇌 역할 (중간 계층)
        self.fc = nn.Linear(hidden_dim, vocab_size) # 최종 결과를 예측해서 뽑아주는 출력기 (출력 계층)

    # 모델 실행 메서드 , forward함수의 결과가 모델의 예측값
    def forward(self, x):
        x = self.embed(x)                                   # 단어 인덱스를 임베딩 벡터로 변환
        x, _ = self.lstm(x.view(len(x), 1, -1))             # LSTM 처리
        out = self.fc(x[-1])                                # 마지막 시간의 출력 → 분류
        return out



vocab_size = len(word2idx)
# 모델 인스턴스 생성 , 이 부분에 입력한 숫자로 모델의 파라미터가 정해짐
model = MiniLSTM(vocab_size, embed_dim=5, hidden_dim=5)
# 총 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📦 총 파라미터 수: {total_params}")
# 예측값과 정답 비교에 사용할 손실함수 설정 ( 손실 함수란 모델의 예측값과 실제 정답 사이의 오차를 수치르 계산하는 함수 , 예측이 틀리면 값이 커지고 맞으면 값이 작아짐 )
loss_fn = nn.CrossEntropyLoss()
#모델 파라미터를 업데이트할 옵티마이저 설정 ( 손실 함수가 계산한 오차를 바탕으로 모델의 가중치를 바꾸는 역할 )
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6️⃣ 학습 전 초기 가중치 일부 출력
print("\n✅ 초기 가중치 샘플 (fc.weight):")
print(model.fc.weight[:3])  # 일부 출력

print("\n🔁 학습 시작")
for epoch in range(10):  # 🔁 전체 학습 10번 반복
    total_loss = 0
    print(f"\n🌀 Epoch {epoch + 1}")

    for i, (x, y) in enumerate(data):  # 🔁 훈련 샘플 순회
        optimizer.zero_grad()

        # 1️⃣ 예측
        pred = model(x)

        # 2️⃣ 손실 계산
        loss = loss_fn(pred, y)

        # 3️⃣ 역전파 (기울기 계산)
        loss.backward()

        # 💡 어떤 단어의 가중치 확인할지 (예: 단어 0번 = idx2word[0])
        target_word_idx = 0
        target_word = idx2word[target_word_idx]
        weight_before = model.fc.weight.data.clone()[target_word_idx][:3]

        print(f"  Sample {i + 1}")
        print(f"    📥 입력 x         : {x.item()} → ({idx2word[x.item()]})")
        print(f"    🎯 정답 y         : {y.item()} → ({idx2word[y.item()]})")
        print(f"    🧠 예측 pred      : {pred.argmax().item()} → ({idx2word[pred.argmax().item()]})")
        print(f"    ❌ 손실(loss)     : {loss.item():.6f}")
        print(f"    🧊 fc.weight[단어 '{target_word}'] (전) : {weight_before}")

        # 4️⃣ 가중치 업데이트
        optimizer.step()

        weight_after = model.fc.weight.data.clone()[target_word_idx][:3]
        print(f"    🔥 fc.weight[단어 '{target_word}'] (후) : {weight_after}")

        total_loss += loss.item()

    print(f"✅ Epoch {epoch + 1} 끝! 평균 손실: {total_loss / len(data):.4f}")


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
