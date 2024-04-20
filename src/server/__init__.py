import os
import sqlite3
from textwrap import indent
from flask import Flask, jsonify, request, current_app, g 
from flask_cors import CORS, cross_origin
from pandas.core import methods
from sklearn.metrics import accuracy_score
import structlog
import json
from functools import wraps
from glob import glob
import base64
import time
from pprint import pprint
from server.db import get_base_learners_for_model
from dataclasses import asdict

from server.data_helpers import OverallPerf, PerfMetric, Run

from server.ids import lccde, mth, treebased
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

    
    # On success, returns a dict containing the valid and type-correct parameters 
    # On failure, returns an error string to provide feedback to frontend
    def validate_and_convert_params(connection: sqlite3.Connection, request_hyperparams: dict, model_name: str) -> dict|str:
        base_learners = db.get_base_learners_for_model(connection, model_name)
        param_dict = dict()
        
        for learner_name, params in request_hyperparams.items(): 
            # verify learners in request exist in db
            if learner_name not in base_learners:
                return f'`{learner_name} is not a base learner used in `{model_name}.'

            truth_params = db.get_hyperparameters_for_learner(connection, learner_name)

            for param_name, pinfo in params.items():
                # validate all hyperparameters exist in the db as an easy way to check obvious errors
                hp = [hp for hp in truth_params if hp.name == param_name]
                if len(hp) == 0: 
                    return f'`{param_name}` is not a valid hyperparameter for `{learner_name}`'
                hp = hp[0]
            # at this point we know all hyperparams are good to go
            ps = { pname: p['v'] for pname, p in params.items() }
            param_dict.update({learner_name: ps})
        return param_dict


    @app.route('/api/run', methods=['GET', 'POST'])
    @require_json_fields(['runid', 'model_name', 'hyperparameters'], methods=['POST'])
    def run(): 
        DB = db.get_db()
        if request.method == 'GET':
            log.info('GET api/run')
            # get every run
            sql = r'''
            SELECT * from Run;
            '''
            def try_convert_datatype(value, hint) -> str|int|float:
                try:
                    if hint == 'int' or hint == 'integer':
                        return int(value)
                    elif hint == 'float':
                        return float(value)
                    elif hint == 'bool':
                        return bool(value)
                    elif hint == 'str' or hint == 'string':
                        return str(value)
                    else:
                        pass # its ok to leave as a string... hopefully!
                except ValueError: 
                    return str(value)

            runs = [
                Run(
                id= r[0],
                timestamp= r[1],
                run_tag=r[2],
                detection_model_name=r[3],
                learner_configuration=dict(),
                learner_overalls=dict(),
                learner_performance_per_attack=dict(),
                dataset=r[4], # TODO: run should store the dataset but doesnt have a column in the table
                confusion_matrices=dict())
             for r in DB.execute(sql).fetchall()] 

            for run in runs:
                base_learners = get_base_learners_for_model(DB, run.detection_model_name)
                for base_learner in base_learners:
                    # get all the learner configs for this run
                    sql = rf'''
                        SELECT LearnerConfig.*, Hyperparameter.name, Hyperparameter.datatype_hint, Hyperparameter.id
                        FROM LearnerConfig
                        JOIN Hyperparameter ON LearnerConfig.hyperparameter_id = Hyperparameter.id
                        WHERE LearnerConfig.run_id = "{run.id}" and LearnerConfig.base_learner_name="{base_learner}";
                    '''
                    config = DB.execute(sql).fetchall()
                    run.learner_configuration.update({
                        base_learner: { str(p[5]): { "id": p[7], "v": try_convert_datatype(str(p[4]), p[6]) } for p in config}
                    })

                    # get all the per-attack metrics
                    sql = rf'''
                        SELECT *
                        FROM AttackPerfMetric
                        JOIN Run ON Run.id=AttackPerfMetric.run_id
                        WHERE AttackPerfMetric.base_learner_name='{base_learner}' and Run.id='{run.id}';
                    '''
                    perfs = DB.execute(sql).fetchall()
                    run.learner_performance_per_attack.update({
                        base_learner: { str(p[6]): PerfMetric(
                            support=p[1],
                            f1_score=p[2],
                            precision=p[3],
                            recall=p[4],
                        ) for p in perfs}

                    })

                    sql = rf'''
                        SELECT *
                        FROM OverallPerfMetric
                        JOIN Run ON Run.id=OverallPerfMetric.run_id
                        WHERE  OverallPerfMetric.base_learner_name='{base_learner}' and Run.id='{run.id}';
                    '''
                    overall_perf = DB.execute(sql).fetchone()
                    run.learner_overalls.update({
                        base_learner: OverallPerf(
                            accuracy=overall_perf[3],
                            macro_avg_precision=overall_perf[3],
                            macro_avg_recall=overall_perf[4],
                            macro_avg_f1_score=overall_perf[5],
                            macro_avg_support=overall_perf[6],
                            weighted_avg_precision=overall_perf[7],
                            weighted_avg_recall=overall_perf[8],
                            weighted_avg_f1_score=overall_perf[9],
                            weighted_avg_support=overall_perf[10],
                    )})

                    sql = rf'''
                        SELECT *
                        FROM ConfusionMatrix
                        JOIN Run ON Run.id= ConfusionMatrix.run_id
                        WHERE ConfusionMatrix.base_learner_name='{base_learner}' and Run.id='{run.id}';
                    '''
                    conf_mats = DB.execute(sql).fetchone()
                    run.confusion_matrices.update({
                        base_learner: conf_mats[4]
                    })
            return jsonify(runs)

        elif request.method == 'POST' and request.json:
            # extract the params from the request
            run_tag = request.json['runid']
            model_name = request.json['model_name'].lower()
            hyperparameters = request.json['hyperparameters'] or dict()
            dataset = request.json['dataset'] or 'CICIDS2017_sample_km.csv' 

            param_dict: dict|str = validate_and_convert_params(DB, hyperparameters, model_name)
            if type(param_dict) == str:
                return jsonify({ 'error': param_dict}), 400

            validated_params: dict = param_dict # concretize the type for the ide

            if model_name == 'lccde':
                before = time.time()
                run = lccde.train_model(run_tag, validated_params, dataset=dataset)
                run.store(db.get_db(), hyperparameters)
                after = time.time()
                dur = (after-before) * 1000
                log.info(f'Training LCCDE took: {dur} ms')
                return jsonify(run)
            elif model_name == 'mth':
                before = time.time()
                run = mth.train_model(run_tag, validated_params, dataset=dataset)
                run.store(db.get_db(), hyperparameters)
                after = time.time()
                dur = (after-before) * 1000
                log.info(f'Training mth took: {dur} ms')
                return jsonify(run)
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
        return jsonify([s[5:] for s in glob('data/*.csv')])

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
