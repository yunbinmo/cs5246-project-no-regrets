from api import process_text
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    post_content = ""  # initialization
    if request.method == 'POST':
        post_content = request.form['postContent']
        # post_content += "loooooool"
        # dummy message for testing
        output = process_text(post_content)
        message = "<b>Sensitive Information Detected:</b><br> Phone: {}<br>Email: {}<br>Name: {}<br>Address: {}<br>Disease: {}<br>Toxic sentence: {}".format(
            '; '.join([key.upper() + ": " + ("None" if len(value) == 0 else (', '.join(value))) for key, value in list(output['sensitive_info'].items())[:3]]), 
            ', '.join(output['sensitive_info']['email']), 
            ', '.join(output['sensitive_info']['name']), 
            ', '.join(output['sensitive_info']['address']),
            ', '.join([str(o) for o in output['sensitive_info_disease']]),
            'Yes (think again before posting!)' if output['toxic'] == 1 else 'No')
    # render processing results
    return render_template('todo.html', message=message, post_content=post_content)

if __name__ == '__main__':
    app.run(debug=True)
