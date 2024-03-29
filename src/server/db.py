### Database related functions go here
### The database is initialized here as well, creating tables and populating tables with data. 

from flask import Flask, jsonify, request, current_app, g
import sqlite3
import csv
from uuid import uuid4
import click
from .ids import lccde, mth, treebased
from dataclasses import dataclass



def get_db() -> sqlite3.Connection:
    if 'db' not in g:
        g.db = sqlite3.connect(current_app.config['DATABASE'], timeout=20)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

    sql = r'''
    INSERT INTO Hyperparameter(id, base_learner_name, name, description, datatype_hint, optional, default_value) VALUES(?, ?, ?, ?, ?, ?, ?)
    '''
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

    # populate LearnsWith relation
    for learner in lccde.BASE_LEARNERS: 
        db.execute(r'''
        INSERT OR IGNORE INTO LearnsWith (id, detection_model_name, base_learner_name) VALUES(?, ?, ?)
                   ''', (str(uuid4()), 'lccde', learner))
    db.commit()


@click.command('init-db')
def init_db_cmd():
    init_db()
    click.echo('Initialized the database')

def init_db_instance(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_cmd)

@dataclass 
class Hyperparameter:
    id: str
    description: str
    name: str
    base_learner_name: str
    datatype_hint: str
    optional: str
    default_value: str

def get_hyperparameters_for_learner(db: sqlite3.Connection, learner: str) -> list[Hyperparameter]:
    sql = rf'''
    SELECT * from Hyperparameter WHERE base_learner_name="{learner}";
    '''
    result: list[Hyperparameter] = []
    for row in db.execute(sql).fetchall():
        result.append(Hyperparameter(*row))

    return result


@dataclass 
class BaseLearner:
    name: str

def get_base_learners_for_model(db: sqlite3.Connection, model: str) -> list[BaseLearner]: 
    sql = rf'''
    SELECT base_learner_name from LearnsWith WHERE detection_model_name="{model}";
    '''
    return [x[0] for x in db.execute(sql).fetchall()]












