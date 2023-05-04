from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


from transformers import AutoTokenizer, AutoModel


model_path = "BelleGroup/BELLE-7B-2M" # You can modify the path for storing the local model
model =  AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Human:")
line = input()
while line:
        inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.35, repetition_penalty=1.2)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("Assistant:\n" + rets[0].strip().replace(inputs, ""))
        print("\n------------------------------------------------\nHuman:")
        line = input()