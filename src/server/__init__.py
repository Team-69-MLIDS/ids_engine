import os
import sqlite3
from flask import Flask, jsonify, request, current_app, g 
from flask_cors import CORS, cross_origin
from pandas.core import methods
import structlog
import json
from functools import wraps
from glob import glob
import base64
import time

from server.ids import lccde
structlog.stdlib.recreate_defaults()
log = structlog.get_logger('main')

def require_json_fields(fields, methods):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.method not in methods:
                return f(*args, **kwargs)

            if not request.is_json:
                return jsonify({'error': 'Request must be JSON.'}), 400

            json_data = request.get_json()

            missing_fields = [field for field in fields if field not in json_data]

            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

            return f(*args, **kwargs)

        return wrapper
    return decorator

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, 'ids.sqlite'),
    )
    CORS(app, supports_credentials=True)

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
                    'id': p[0],
                    'name': p[2],
                    'description': p[1],
                    'optional': p[5],
                    'default': p[6],
                    'type_hint': p[4],
                    })
            return jsonify(response)

        response = dict()
        learners = con.execute('SELECT DISTINCT base_learner_name from hyperparameter;').fetchall()
        for p in learners:
            response.update({ p[0]: []})
    
        print(response)

        params = con.execute('SELECT name, base_learner_name, description, optional, default_value, datatype_hint, id from hyperparameter;').fetchall()
        for p in params: 
            response[p[1]].append({
                'id': p[6],
                'name': p[0],
                'description': p[2],
                'optional': p[3],
                'default': p[4],
                'type_hint': p[5]
                })
            # print(p[0])
        return jsonify(response)


    @app.route('/api/run', methods=['GET', 'POST'])
    @require_json_fields(['runid', 'model_name', 'hyperparameters'], methods=['POST'])
    def run(): 
        DB = db.get_db()
        if request.method == 'GET':
            log.info('GET api/run')
        elif request.method == 'POST':
            # extract the params from the request
            run_tag = request.json['runid']
            model_name = request.json['model_name'].lower()
            hyperparameters = request.json['hyperparameters']
            if 'dataset' in request.json:
                dataset = request.json['dataset'] 
            else: 
                dataset = 'CICIDS2017_sample_km.csv' # dataset is optional 


            if model_name == 'lccde':
                # source of truth
                base_learners = db.get_base_learners_for_model(DB, 'lccde')
                param_dict = dict()
                
                for learner_name, params in hyperparameters.items(): 
                    # verify learners in request exist in db
                    if learner_name not in base_learners:
                        return jsonify({'error': f'`{learner_name} is not a base learner used in `{model_name}.'}), 400


                    requested_params = {x['parameter_name']: x['value'] for x in params}
                    print(requested_params)
                    truth_params = db.get_hyperparameters_for_learner(DB, learner_name)

                    for param_name, value in requested_params.items():
                        # validate all hyperparameters exist in the db as an easy way to check obvious errors
                        hp = [hp for hp in truth_params if hp.name == param_name]
                        if len(hp) == 0:
                            return jsonify({'error': f'`{param_name}` is not a valid hyperparameter for `{learner_name}`'}), 400

                        hp = hp[0]
                        # do basic type conversion, defaulting to string if there is not a simple/known type 
                        old_val = value

                        try:
                            if hp.datatype_hint == 'int' or hp.datatype_hint == 'integer':
                                requested_params[param_name] = int(value)
                            elif hp.datatype_hint == 'float':
                                requested_params[param_name] = float(value)
                            elif hp.datatype_hint == 'bool':
                                requested_params[param_name] = bool(value)
                            elif hp.datatype_hint == 'str' or hp.datatype_hint == 'string':
                                requested_params[param_name] = str(value)
                            else:
                                pass # its ok to leave as a string... hopefully!
                        except ValueError: 
                            # revert 
                            requested_params[param_name] = str(old_val)
                        param_dict.update({learner_name: requested_params})
                

                    log.info('requested_params: ', param_dict)
                before = time.time()
                run = lccde.train_model(run_tag, param_dict,  dataset=dataset)
                run.store(db.get_db())
                after = time.time()
                dur = (after-before) * 1000
                log.info(f'Training LCCDE took: {dur} ms')
                return jsonify(run)
            elif model_name == 'mth':
                # run = mth.train_model(run_tag, learner_configuration_map={})
                pass
            elif model_name == 'treebased':
                # run = treebased.train_model(run_tag, learner_configuration_map={})
                pass


            # TODO(tristan): store the `run` 
            log.info('POST api/run')
        return jsonify("run!")

    @app.route('/api/model_names',  methods=["GET"])
    @cross_origin(supports_credentials=True)
    def models(): 
        sql = r'''
        SELECT * from DetectionModel;
        '''
        DB = db.get_db()
        algs = DB.execute(sql).fetchall()
        return [x[0] for x in algs]

    @app.route('/api/datasets',  methods=["GET"])
    def datasets(): 
        return jsonify(glob('data/*.csv'))

    @app.route('/api/test_confusion_matrix')
    def be_confused():
        if request.method == 'GET':
            with open('xgboost_confusion_matrix.png', 'rb') as f:
                data = f.read()
                return jsonify({
                               'b64_image': base64.b64encode(data).decode('utf-8'),
                               'info': "you can either convert this image into a actual DOM javascript image by converting it back to bytes and feeding the bytes to the image constructor, or you can directly embed the base64 in the image tag. Ping me if u want help"
                })
        else:
            return jsonify("use GET"), 400




    return app
