from flask import Flask, request, render_template
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)

# Load the GPT-Neo model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        prompt = request.form["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        return render_template("index.html", prompt=prompt, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
