import os
import sqlite3
from types import MethodDescriptorType
from flask import Flask, jsonify, request, current_app, g 
from flask_cors import CORS
from pandas.core import methods
import structlog
import json
from functools import wraps


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

    @app.route('/api/algorithm_names')
    def hello():
        con = db.get_db()

        algorithm_names = con.execute('SELECT DISTINCT name FROM DetectionModel;').fetchall()

        names = []
        for name in algorithm_names:
            names.append(name[0])

        dict = {
            "algorithms": names
        }

        return dict


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

    def generate_default_run_id():
        return "default run id" 


    import datetime
    from dateutil.parser import parse

    @app.route('/api/run', methods=['GET', 'POST'])
    @require_json_fields(['runid', 'model_name', 'hyperparameters'], methods=['POST'])
    @require_json_fields(['runid'], methods=['GET'])
    def run(): 
        if request.method == 'GET':
            log.info('GET api/run')

            con = db.get_db()

            fromV = request.args.get('from')
            till = request.args.get('till')
            runid = request.args.get('runid')

            if fromV is not None:
                fromTimestamp = parse(fromV, None).strftime("%Y-%m-%d %H:%M:%S")
            if till is not None:
                tillDate = parse(till, None)
                tillDateTimestamp = tillDate.strftime("%Y-%m-%d %H:%M:%S")
                tillDateNoTime = tillDate.strftime("%Y-%m-%d 00:00:00")

                if tillDateTimestamp == tillDateNoTime:
                    tillTimestamp = tillDate + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
                else:
                    tillTimestamp = tillDate

            if runid is not None:
                if fromV is not None and till is not None:
                    sql = "SELECT * FROM run WHERE (timestamp BETWEEN '" + fromTimestamp + "' AND '" + tillTimestamp + "') AND id = '"+runid+"';"
                elif fromV is not None:
                    sql = "SELECT * FROM run WHERE (timestamp >= '" + fromTimestamp + "') AND id = '"+runid+"';"
                elif till is not None:
                    sql = "SELECT * FROM run WHERE (timestamp <= '" + tillTimestamp + "') AND id = '"+runid+"';"
                else:
                    sql = "SELECT * FROM run WHERE id = '"+runid+"';"
            else:
                raise Exception("No run id provided. Please provide one")

            receivedRuns = con.execute(sql).fetchall()
            runs = []

            for run in receivedRuns:
                algorithm = run[3]
                runId = run[0]
                timestamp = run[1]

                model_name = run[3]
                confusion_matrix = [1,1,1,1,1]

                sql2 = "SELECT * FROM PrefMetric WHERE run_id = '"+ runid +" LIMIT 1';"
                prefMetric = con.execute(sql2).fetchall()[0]

                f1_score = prefMetric[2]
                precision = prefMetric[3]
                recall = prefMetric[5]
                macro_avg = prefMetric[6]
                weighted_avg = prefMetric[7]

                dict = {
                    "algorithm": algorithm,
                    "runid": runId,
                    "timestamp": timestamp,
                    "enemble": [
                        {
                            "model_name": model_name,
                            "confusion_matrix": [confusion_matrix],
                            "f1_score": f1_score,
                            "precision": precision,
                            "recall": recall,
                            "macro_avg": macro_avg,
                            "weighted_avg": weighted_avg
                        }
                    ]
                }
                runs.append(dict)

            return runs

        elif request.method == 'POST':
            # extract the params from the request
            run_tag = request.json['runid']
            model_name = request.json['model_name']
            hyperparameters = request.json['hyperparameters']
            print(run_tag, model_name, json.dumps(hyperparameters, sort_keys=True, indent=2), sep='\n' )

            # TODO(tristan): validate each learner in hyperparameters exists
            # TODO(tristan): LOW PRIO validate all hyperparameters exist in the db as an easy way to check obvious errors
            run = lccde.train_model(run_tag, learner_configuration_map={})

            # TODO(tristan): store the `run` 
            log.info('POST api/run')
        return jsonify("run!")

    return app
