from api import process_text
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""  # 初始化空消息
    if request.method == 'POST':
        post_content = request.form['postContent']
        # 在这里处理post_content
        # post_content += "loooooool"
        print(post_content)
        message = process_text(post_content)  # 模拟处理后的消息
        # message = "数据已接收: " + process_text(post_content)  # 模拟处理后的消息
    # 返回到HTML页面，带有处理后的消息
    return render_template('todo.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
