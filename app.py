from flask import Flask,render_template,request
import caption
app = Flask(__name__)


@app.route('/')
def start():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def image_captioing():
    if request.method=="POST":
        f = request.files["image"]  
        path = "./static/{}".format(f.filename)
        f.save(path)
        data = caption.Caption(path)
        result_dic={
        'image':path,
        'val':data 
        }
    return render_template("index.html",my_caption = result_dic)


if __name__=="__main__":
    app.debug = True
    app.run()