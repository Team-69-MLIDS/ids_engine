from flask import Flask, jsonify, request
import sys
import sqlite3
import csv
from uuid import uuid4
import ids

app = Flask("idsendpoint")

@app.route('/', methods=['GET'])
def test_req():
    if request.method == 'GET': 
        data = {
            'test_value': 'hello',
            'test_int': 1923801,
        }
        return jsonify(data)
    else: 
        return jsonify({'msg': 'only GET is supported'})

def db_init(db: sqlite3.Connection):
    with open('./src/dbinit.sql') as dbinit:
        init: str = dbinit.read()
        # creates tables and some enum tables according to dbinit.sql
        db.executescript(init)

    sql = 'INSERT INTO Hyperparameter(id, MachineLearningLibrary_name, name, description, datatype_id) VALUES(?, ?, ?, ?, ?)'
    # import hyperparameters
    with open('xgboost.csv') as xgboost_params:
        reader = csv.reader(xgboost_params)
        next(reader, None) # skip header row
        for row in reader:
            db.execute(sql, (str(uuid4()), "XGBoost", row[1], row[3], "string"))
            db.commit()


if __name__=='__main__':
    with sqlite3.connect('idsdb.db', timeout=20) as db:
        db_init(db)
        app.run(debug=True)
