
# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/flan-t5-xl-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/flan-t5-xl-squad2")

question, text = "Which company produces the iPod?", "The iPod is a line of portable media players and multi-purpose pocket computers designed and marketed by Apple Inc. The first line was released on October 23, 2001, about 8½ months after iTunes (Macintosh version) was released. The most recent iPod redesigns were announced on July 15, 2015. There are three current versions of the iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch."

inputs = tokenizer(question, text, return_tensors="pt")
outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))