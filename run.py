import gradio as gr
import openai
import json
import pandas as pd
import time
from datetime import datetime

def generate(system_text, text, model):
    messages = [
        {"role": "system", "content": f"{system_text}"},
        {"role": "user", "content": f"{text}"}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    print(json.dumps(response, ensure_ascii=False, indent=2))
    return response.choices[0].message.content

def upload_file(model, file):
    openaiFile = openai.File.create(
        file=open(file.name, "rb"),
        purpose='fine-tune'
    )
    fileUploadResult = json.dumps(openaiFile, ensure_ascii=False)
    yield [fileUploadResult, '等待上传完成开始创建训练任务', '']
    if openaiFile.status == 'uploaded':
        while True:
            try:
                fineTuningJob = openai.FineTuningJob.create(training_file=openaiFile.id, model=model)
                if fineTuningJob.status == 'created':
                    fineTuningJobResult = "开始训练，预计需要一定时间...\n" + json.dumps(fineTuningJob, ensure_ascii=False)
                    yield [fileUploadResult, fineTuningJobResult, '']
                    jobMsg = ''
                    while True:
                        jobResult = openai.FineTuningJob.retrieve(fineTuningJob.id)
                        jobMsg += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + json.dumps(jobResult, ensure_ascii=False, separators=(',', ':')) + "\n"
                        jobMsg = '\n'.join(jobMsg.split('\n')[-5:])
                        yield [fileUploadResult, fineTuningJobResult, jobMsg]
                        if (jobResult.status == 'succeeded'):
                            return [fileUploadResult, f"训练完成\n任务id：{jobResult.id}\n模型：{jobResult.fine_tuned_model}\n训练tokens：{jobResult.trained_tokens}", jobMsg]
                        time.sleep(10)
                break
            except Exception as e:
                yield [fileUploadResult, "文件上传中，请等待...\n" + str(e), '']
                time.sleep(10)
    else:
        return ['上传文件失败', '', '']
    return ['', '', '']

def upload_excel(sys_content, file):
    df = pd.read_excel(file.name)
    msg = '';
    for index, row in df.iterrows():
        msg += '{"messages": [{"role": "system", "content": "'+str(sys_content).replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\'")+'"}, {"role": "user", "content": "'+str(row[0]).replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\'")+'"}, {"role": "assistant", "content": "'+str(row[1]).replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\'")+'"}]}\n'
    with open('tmp.jsonl', 'w') as f:
        f.write(msg)
    return 'tmp.jsonl'

examples = [
    ["你是一个训练的新模型", "测试问题", "gpt-3.5-turbo"]
]

model_arr = [
    "gpt-3.5-turbo",
    "gpt-4",
    "ft:gpt-3.5-turbo-0613:inagora::7ugHhB7d",
]

with gr.Blocks() as demo:

    gr.Markdown("内容生成器")

    with gr.Tab("内容生成"):
        text_input = [gr.Textbox(lines=5, label="system-内容"), gr.Textbox(lines=5, label="user-内容"), gr.Dropdown(
                model_arr, label="模型", info="请选择一个模型", value="gpt-3.5-turbo"
            )
        ]
        text_output = gr.Textbox(label="输出内容")
        text_button = gr.Button("生成")
        gr.Examples(examples=examples, inputs=text_input)
        text_button.click(generate, inputs=text_input, outputs=text_output)
    
    with gr.Tab("微调"):
        input = [gr.Radio(["gpt-3.5-turbo-0613", "babbage-002", "davinci-002"], label="微调模型", info="请选择微调基于的模型", value="gpt-3.5-turbo-0613"), gr.File(label="训练文件", file_types=['xlsx', 'csv'])]
        output = [gr.Textbox(label="文件上传结果"), gr.Textbox(label="训练结果"), gr.Textbox(label="训练日志")]
        text_button = gr.Button("开始微调")
        text_button.click(upload_file, inputs=input, outputs=output)

    with gr.Tab("生成微调文件"):
        input = [gr.Textbox(label="system提示内容", lines=5), gr.File(label="训练文件", file_types=['xlsx', 'csv'])]
        output = gr.File(label="jsonl训练文件")
        text_button = gr.Button("生成")
        text_button.click(upload_excel, inputs=input, outputs=output)

demo.queue().launch()