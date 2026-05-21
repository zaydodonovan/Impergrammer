import Helper as H

try:
    from transformers import pipeline
except Exception as e:
    H.Install()
    from transformers import pipeline

def paraphrase_text(input_text, style_reference):
    generator = pipeline(task="text2text-generation", model="t5-small", tokenizer="t5-small")
    prompt = f"paraphrase: {input_text} in the style of: {style_reference[:200]}"
    paraphrased_text = generator(prompt, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']
    return paraphrased_text.strip()

def main():
    H.Startup()
    file_name = "Data.txt"

    try:
        with open(file_name, 'r') as file:
            original_text = file.read()
    except FileNotFoundError:
        print(f"File '{file_name}' Not Found.")
        return
    except Exception as e:
        print(f"Error Occurred While Reading The File: {e}")
        return

    while True:
        try:
            user_input = input("Text To Rephrase: ")

            if user_input.strip() in ("", " ", "99"):
                H.clear()
                break
            elif user_input.strip() in ("clear", "cls"):
                H.clear()
                H.banner()
            else:
                paraphrased_text = paraphrase_text(user_input, original_text)
                print("\nParaphrased Text:")
                print(f"{paraphrased_text}\n")

        except KeyboardInterrupt:
            H.clear()
            exit()

if __name__ == "__main__":
    main()