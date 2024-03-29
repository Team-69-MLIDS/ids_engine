import os
import sqlite3
from flask import Flask, jsonify, request, current_app, g 
from flask_cors import CORS
import structlog

from server.ids import lccde
structlog.stdlib.recreate_defaults()
log = structlog.get_logger('main')

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, 'ids.sqlite'),
    )
    cors = CORS(app)

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

    @app.route('/api/hyperparameters')
    def hyperparameters():
        con = db.get_db()

        model = request.args.get('model')
        if model is not None: 
            log.info(f'finding hyperparameters for {model}')
            sql = rf'''SELECT h.* 
            FROM Hyperparameter h
            JOIN LearnsWith lw ON h.base_learner_name = lw.base_learner_name
            WHERE lw.detection_model_name = "{model}";'''
            learners = con.execute(sql).fetchall()
            response = dict()
            for p in learners:
                base_learner_name = p[3]

                if base_learner_name not in response:
                    response.update({base_learner_name: []})

                response[base_learner_name].append({
                    'name': p[2],
                    'description': p[1],
                    'optional': p[5],
                    'default': p[6]
                    })
            return jsonify(response)

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
        return jsonify(response)

    def generate_default_run_tag():
        return "default run id" 

    @app.route('/api/run', methods=['GET', 'POST'])
    def run(): 
        if request.method == 'GET':
            log.info('GET api/run')
        elif request.method == 'POST':
            # extract the params from the request
            run_tag = request.args.get('run_tag') or generate_default_run_tag()
            lccde.train_model(run_tag, learner_configuration_map={})
            log.info('POST api/run')
        return jsonify("run!")

    return app
