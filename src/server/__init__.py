import os
import sqlite3
from flask import Flask, jsonify, request, current_app, g

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, 'ids.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    from . import db
    db.init_db_instance(app)

    # a simple page that says hello
    @app.route('/api/hyperparameters')
    def hello():
        con = db.get_db()
        params = con.execute('SELECT name from hyperparameter').fetchall()

        response = []
        for p in params: 
            print(p[0])
            response.append(p[0])
        return response


    return app
