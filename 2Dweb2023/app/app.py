import os
from flask import *
from werkzeug.utils import secure_filename
from funtion import *

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "static\\data"
app.secret_key = 'This is your secret key to utilize session in Flask'

#route to home
@app.route('/')
def index():
    return render_template("index.html")

#route to csv upload page
@app.route('/input', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')
        
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        if data_filename == "":
            return render_template("input.html", data="No file selected")
        f.filename = "cal.csv"

        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        session['uploaded_data_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('input2.html')
    return render_template("input.html")

#route to show csv table
@app.route('/excel', methods=['GET', 'POST'])
def showData():
    if len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0:
        return render_template('excel.html', data_var=None)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)

    if data_file_path is None:
        return render_template('excel.html', data_var=None)
    
    inx = run_idx(data_file_path)
    convertcsv(inx)
    return render_template('excel.html', data_var=inx )

#route to download prediction csv
@app.route('/download')
def download():
    file_path = "static\\csv\\Data.csv"
    return send_file(file_path, as_attachment=True)

#route to about
@app.route('/about')  
def aboutpage():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
