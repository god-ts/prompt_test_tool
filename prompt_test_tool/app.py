from flask import Flask, request, render_template
import requests
import time
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
from concurrent.futures import ThreadPoolExecutor



OPENAI_API_KEY = 'sk-Nothing'
df = pd.read_csv('data/embedding信息库_挖需_question表_问题embedding.csv', header=0, names=['问题编号', 'question', 'factid', 'category', 'fact', 'answer', '备注', 'vector'])
vectors = df['vector'].apply(eval).apply(np.array)

def get_sorted_indices(numbers):
    indexed_numbers = list(enumerate(numbers))
    sorted_indices = sorted(indexed_numbers, key=lambda x: x[1])
    return [index for index, number in sorted_indices]

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    url = "http://c.iaiapp.com:3500/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model
    }
    time0 = time.time()
    while True:
        if time.time() - time0 >= 30:
            return "超时"
        try:
            response = requests.post(url, headers=headers, json=payload)
            response_data = response.json()
            return response_data['data'][0]['embedding']
        except Exception as e:
            print(e)

def chat_gpt(message: list, model='gpt-3.5-turbo'):
    if model == 'gpt-4':
        url = "http://c.iaiapp.com:4000/v1/chat/completions"
    else:
        model = 'gpt-3.5-turbo'
        url = "http://c.iaiapp.com:3500/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": message,
        "temperature": 0
    }
    time0 = time.time()
    while True:
        if time.time() - time0 >= 90:
            return "请求超时"
        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()['choices'][0]['message']['content'].strip()
            return result
        except Exception as e:
            print(e)
            print(message)

history_list_35 = []
history_list_40 = []
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    global history_list_35
    global history_list_40
    if request.method == 'POST':
        inputs = [request.form.get(f'input{i}') for i in range(1, 11)]
        user_input = request.form.get('user-input')
        temperature = request.form.get('temperature')
        ordering_system = request.form.get('ordering_system')
        ordering = request.form.get('ordering')
        result_system35 = '\n'.join([inputs[int(i)-1] for i in ordering_system])
        result_system40 = '\n'.join([inputs[int(i) - 1] for i in ordering_system])
        result_dialogue35 = '\n'.join([inputs[int(i) - 1] for i in ordering])
        result_dialogue40 = '\n'.join([inputs[int(i) - 1] for i in ordering])

        input_embedding = get_embedding(user_input)
        distances = distances_from_embeddings(input_embedding, vectors, distance_metric='cosine')
        fact = ""
        qa = ""
        history_35 = "\n".join(history_list_35)
        history_40 = "\n".join(history_list_40)
        for i in get_sorted_indices(distances)[:1]:
            if 1 - distances[i] >= 0.9:
                fact = df['fact'][i]
                qa = user_input + "\n" + df['answer'][i]

        result_system35 = result_system35.replace('{embedding}', fact)
        result_system35 = result_system35.replace('{userinput}', user_input)
        result_system35 = result_system35.replace('{example}', qa)
        result_system35 = result_system35.replace('{history}', history_35)

        result_system40 = result_system40.replace('{embedding}', fact)
        result_system40 = result_system40.replace('{userinput}', user_input)
        result_system40 = result_system40.replace('{example}', qa)
        result_system40 = result_system40.replace('{history}', history_35)

        result_dialogue35 = result_dialogue35.replace('{embedding}', fact)
        result_dialogue35 = result_dialogue35.replace('{userinput}', user_input)
        result_dialogue35 = result_dialogue35.replace('{example}', qa)
        result_dialogue35 = result_dialogue35.replace('{history}', history_35)

        result_dialogue40 = result_dialogue40.replace('{embedding}', fact)
        result_dialogue40 = result_dialogue40.replace('{userinput}', user_input)
        result_dialogue40 = result_dialogue40.replace('{example}', qa)
        result_dialogue40 = result_dialogue40.replace('{history}', history_40)


        message35 = []
        message35.append({"role": "system", "content": result_system35})
        message35.append({"role": "user", "content": result_dialogue35})

        message40 = []
        message40.append({"role": "system", "content": result_system40})
        message40.append({"role": "user", "content": result_dialogue40})

        #异步
        with ThreadPoolExecutor(max_workers=2) as executor:
            future35 = executor.submit(chat_gpt, message35, 'gpt-3.5-turbo')
            future40 = executor.submit(chat_gpt, message40, 'gpt-4')

        result35 = future35.result()
        result40 = future40.result()

        history_list_35.append(user_input)
        history_list_35.append(result35)

        history_list_40.append(user_input)
        history_list_40.append(result40)


        result =  render_template('index.html', inputs=inputs, middle_result=message35, output_result=message40, user_input=user_input, temperature=temperature, ordering_system=ordering_system, ordering=ordering, middle_result2=result35, output_result2=result40)

        return result
    else:
        history_list_35 = []
        history_list_40 = []
        default_inputs = ['' for _ in range(10)]
        default_result = ''
        default_user_input = ''
        default_temperature = 0
        default_ordering_system = ''
        default_ordering = ''

        return render_template('index.html', inputs=default_inputs, middle_result=default_result, output_result=default_result, user_input=default_user_input, temperature=default_temperature, default_ordering_system=default_ordering_system, ordering=default_ordering)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8786)
