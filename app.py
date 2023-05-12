from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import jieba
import random
import sqlite3
import datetime
from generate import gen

app = Flask(__name__)
app.debug = True


# 首页
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/square', methods=['GET'])
def to_square():
    """跳转到广场页面,并从数据库中读取诗歌"""
    # 连接数据库
    query = request.args.get('query', '')
    conn = sqlite3.connect('poems.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 查询所有诗歌
    sql = "SELECT * FROM poems"
    # 如果有查询条件，则添加模糊查询条件
    if query:
        sql += " WHERE poem LIKE '%{}%'".format(query)
    c.execute(sql)
    poems_db = c.fetchall()

    # 关闭连接
    conn.close()

    # 将诗歌转换为具有所需属性的字典列表
    poems = []
    for poem_db in poems_db:
        poem = {
            'id': poem_db[0],
            'author': poem_db[1],
            'title': poem_db[2],
            'content': poem_db[3],
            'theme': poem_db[4],
            'date': poem_db[5],
            'img_path': poem_db[6]
        }
        poems.append(poem)
    return render_template('square.html', poems=poems)


@app.route('/promote', methods=['GET'])
def promote():
    """读取result.txt然后分词，随机返回一个词语"""
    words = []
    # 如果不存在promotes.txt，则从result.txt中读取并分词
    if not os.path.exists('promotes.txt'):
        with open('result.txt', 'r', encoding='gbk') as f:
            for line in f.readlines():
                # 去除"训练损失"
                line = line.replace('训练损失', '')
                # 用jieba分词,去除标点符号和数字仅保留中文
                words += [word for word in jieba.cut(line) if word.isalpha()]
        # 将分词结果写入promotes.txt
        with open('promotes.txt', 'w', encoding='gbk') as f:
            for word in words:
                f.write(word + '\n')
    # 从promotes.txt中读取词语并随机返回一个
    with open('promotes.txt', 'r', encoding='gbk') as f:
        words = f.readlines()
        res = random.choice(words)
    return jsonify({'promote': res})


# 首页输入完后点击生成调用的函数
@app.route('/predict', methods=['POST'])
def predict():
    # 接受上传的文件
    words = request.form.get('words')
    action = request.form.get('action')
    action = 1 if action == '生成诗词' else 2
    if not words:
        return
    # 调用模型
    try:
        rs = gen(words, action)
    except:
        rs = ['无法根据提示词【%s】写诗，请换一个提示词试试' % words]
    # 随机返回一个static/images下的图片路径
    img_path = "/static/image/%s" % random.choice(os.listdir('./static/image'))
    # 返回poem.html，把gen返回的结果传过去
    return render_template('poem.html', rs=rs, img_path=img_path)


@app.route('/submit_poem', methods=['POST'])
def submit_poem():
    """诗歌生成页面-提交诗歌"""
    author = request.form['author']
    title = request.form['title']
    poem = request.form['poem']
    theme = request.form['theme']
    img_path = request.form['img_path']
    update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 连接数据库
    conn = sqlite3.connect('poems.db')
    c = conn.cursor()

    # 如果没有 poems 表，创建一个
    c.execute('''CREATE TABLE IF NOT EXISTS poems
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, title TEXT, poem TEXT, theme Text, img_path Text, update_time TEXT)''')

    # 将诗歌插入数据库
    c.execute("INSERT INTO poems (author, title, poem, theme, img_path, update_time) VALUES (?, ?, ?, ?, ?,?)",
              (author, title, poem, theme, img_path, update_time))

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

    return redirect(url_for('to_square'))


@app.route('/view_poem/<int:poem_id>')
def view_poem(poem_id):
    """根据id查询诗歌"""
    # 连接数据库
    conn = sqlite3.connect('poems.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # 查询诗歌
    c.execute("SELECT * FROM poems WHERE id=?", (poem_id,))
    poem_db = c.fetchone()
    # 关闭连接
    conn.close()
    # 将诗歌转换为具有所需属性的字典列表
    poem = {
        'id': poem_db['id'],
        'author': poem_db['author'],
        'title': poem_db['title'],
        'content': poem_db['poem'],
        'theme': poem_db['theme'],
        'date': poem_db['update_time'],
        'img_path': poem_db['img_path']
    }
    return render_template('poem_detail.html', poem=poem)


if __name__ == '__main__':
    app.run(debug=True)
