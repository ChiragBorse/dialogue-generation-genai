"""
Question: Text & Dialogue Generation using Pretrained Generative AI Models
Model: microsoft/DialoGPT-medium
Framework: PyTorch & Hugging Face Transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_chatbot():
    model_name = "microsoft/DialoGPT-medium"

    print("Loading pretrained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("\n" + "=" * 50)
    print("English Chatbot is ready! Type 'exit' to quit.")
    print("=" * 50 + "\n")

    chat_history_ids = None

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Bot: Goodbye! Have a great day.")
                break

            # Encode input and append end-of-string token
            new_user_input_ids = tokenizer.encode(
                user_input + tokenizer.eos_token, return_tensors="pt"
            )

            # Append input to conversation history for multi-turn tracking
            bot_input_ids = (
                torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                if chat_history_ids is not None
                else new_user_input_ids
            )

            # Generate response using top-k and nucleus (top-p) sampling
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.75,
            )

            # Slice out generated tokens (excluding past conversation prompt)
            response_tokens = chat_history_ids[:, bot_input_ids.shape[-1] :]
            bot_response = tokenizer.decode(
                response_tokens[0], clean_up_tokenization_spaces=False
            )

            print(f"Bot: {bot_response.strip()}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    run_chatbot()
