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

        response = dict()
        
        learners = con.execute('SELECT DISTINCT base_learner_name from hyperparameter;').fetchall()
        for p in learners:
            response.update({ p[0]: []})
    
        print(response)

        params = con.execute('SELECT name, base_learner_name, description, optional, default_value from hyperparameter;').fetchall()
        for p in params: 
            response[p[1]].append({
                'name': p[0],
                'description': p[2],
                'optional': p[3],
                'default': p[4]
                })
            # print(p[0])
        return response


    return app
