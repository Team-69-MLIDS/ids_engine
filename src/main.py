################ DEPRECATED DONT USE ################
from flask import Flask, jsonify, request, current_app, g
import sqlite3
import csv
from uuid import uuid4
import ids
import structlog


#setup logging
structlog.stdlib.recreate_defaults()
log = structlog.get_logger('ids_api')

app = Flask("idsendpoint")

@app.route('/api/datasets', methods=['GET'])
def datasets():
    if request.method == 'GET': 
        return jsonify('TODO')
    else: 
        return jsonify({'msg': 'only GET is supported'})

@app.route('/api/models', methods=['GET'])
def models():
    '''
    TODO update api documentation to reflect updated terminology
    '''
    if request.method == 'GET': 
        return jsonify('TODO')
    else: 
        return jsonify({'msg': 'only GET is supported'})

@app.route('/api/run', methods=['GET', 'POST'])
def run():
    if request.method == 'GET': 
        # Get Runs stored in the database. 
        # Query parameters can be used to filter the results on date, time frame, and run id.
        return jsonify('TODO')

    if request.method == 'POST':
        # Submit hyperparameters and train the selected algorithm
        return jsonify('TODO')

    else: 
        return jsonify({'msg': 'only GET and POST are supported'})

@app.route('/api/hyperparameters', methods=['GET'])
def hyperparameters():
    if request.method == 'GET': 
        # TODO: return list of hyperparameters
        # TODO: add API doc
        return jsonify('TODO')
    else: 
        return jsonify({'msg': 'only GET is supported'})


def db_initialized(db: sqlite3.Connection) -> bool:
    return len(db.execute("SELECT name from sqlite_master WHERE type='table' AND name='init';").fetchall()) > 0


def db_init():
    db = get_db()
    if db_initialized(db):
        log.info("Database already initialized.")
        return

    log.info('Initializing Database...')
    with open('./src/dbinit.sql') as dbinit:
        init: str = dbinit.read()
        # creates tables and some enum tables according to dbinit.sql
        db.executescript(init)
        db.commit()

    sql = 'INSERT INTO Hyperparameter(id, base_learner_name, name, description, datatype_hint, optional, default_value) VALUES(?, ?, ?, ?, ?, ?, ?)'
    # import hyperparameters
    with open('hyperparams.csv') as xgboost_params:
        reader = csv.reader(xgboost_params)
        next(reader, None) # skip header row
        for row in reader:
            base_learner_name = row[0]
            name = row[1] 
            description = row[4]
            default = row[3]
            optional = row[5]
            datatype_hint = row[2]
            db.execute(sql, (str(uuid4()), base_learner_name, name, description, datatype_hint, optional, default))
            db.commit()
    log.info('Finished initalizing db.')

if __name__=='__main__':
        db_init()
        
        app.run(debug=True)
        close_db()
