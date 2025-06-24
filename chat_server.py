from flask import Flask, request, render_template
from llm_demo import model, word2idx, idx2word
import torch

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    result = ""
    user_input = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input in word2idx:
            with torch.no_grad():
                x = torch.tensor([word2idx[user_input]])
                out = model(x)
                pred_idx = torch.argmax(out).item()
                result = idx2word[pred_idx]
        else:
            result = "⚠️ 단어 사전에 없습니다. (예: 나는, 사과를, 먹었다)"
    return render_template("chat.html", user_input=user_input, result=result)

if __name__ == "__main__":
    app.run(debug=True)
