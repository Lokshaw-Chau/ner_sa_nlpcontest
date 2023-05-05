import openai

openai.api_key
resonpse = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content":"接下来请用简短的回答进行情感分析"},
        {"role": "user", "content": "判断下面这句话的情感倾向：炸了，就2000.浦发没那么好心，草。用0表示负向，1表示正向，2表示中立。例如：情感倾向为1"}
    ]
)

print(resonpse)
