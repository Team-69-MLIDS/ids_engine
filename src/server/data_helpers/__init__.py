from typing import Any
from dataclasses import dataclass, asdict
import base64
import io
import sqlite3
import json
from uuid import uuid4

from pandas.io.formats.style_render import uuid4

@dataclass
class PerfMetric:
    support: float
    f1_score: float
    precision: float
    recall: float

@dataclass
class OverallPerf: 
    accuracy: float
    macro_avg_precision: float
    macro_avg_recall: float
    macro_avg_f1_score: float
    macro_avg_support: float
    weighted_avg_precision: float
    weighted_avg_recall: float
    weighted_avg_f1_score: float
    weighted_avg_support: float

@dataclass
class Run: 
    id: str
    detection_model_name: str
    run_tag: str
    learner_performance_per_attack: dict[str, dict[str, PerfMetric]] # attack class -> PerfMetric 
    learner_configuration: dict[str, dict[str, str|int|float|bool]]
    learner_overalls: dict[str, OverallPerf]
    timestamp: str
    confusion_matrices: dict[str, str]
    dataset: str

    # `config_dict` is straight from request.json['hyperparameters']
    # and contains the id for storing in the db
    def store(self, db: sqlite3.Connection, config_dict: dict): 
        # store the Run
        sql = r'''
        INSERT INTO Run(id, timestamp, run_tag, detection_model_name, dataset) VALUES(?, ?, ?, ?);
        '''
        db.execute(sql, (
            self.id,
            self.timestamp, 
            self.run_tag,
            self.detection_model_name,
            self.dataset
        ))
        

        # store the learner config
        sql = r'''
        INSERT INTO LearnerConfig(config_id, hyperparameter_id, base_learner_name, run_id, value) VALUES(?, ?, ?, ?, ?);
        '''
        for base_learner, hypers in config_dict.items():
            for param_name, param_info in hypers.items():
                db.execute(sql, (
                    str(uuid4()),
                    param_info['id'],
                    base_learner,
                    self.id,
                    param_info['v'],
                ))

        # store overall perf for each learner
        sql = r'''
        INSERT INTO OverallPerfMetric(
            id                        ,
            run_id                    ,
            base_learner_name         ,
            accuracy                  ,
            macro_avg_precision       ,
            macro_avg_recall          ,
            macro_avg_f1_score        ,
            macro_avg_support         ,
            weighted_avg_precision    ,
            weighted_avg_recall       ,
            weighted_avg_f1_score     ,
            weighted_avg_support      
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        for learner, overall in self.learner_overalls.items():
            db.execute(sql, (
                str(uuid4()),
                self.id,
                learner, 
                overall.accuracy                  ,
                overall.macro_avg_precision       ,
                overall.macro_avg_recall          ,
                overall.macro_avg_f1_score        ,
                overall.macro_avg_support         ,
                overall.weighted_avg_precision    ,
                overall.weighted_avg_recall       ,
                overall.weighted_avg_f1_score     ,
                overall.weighted_avg_support      ,
        ))
        sql = r'''
        INSERT INTO AttackPerfMetric(
            id                    , 
            support               , 
            f1_score              , 
            precision             ,
            recall                , 
            base_learner_name     , 
            attack_classification , 
            run_id                 
        )
        VALUES(?,?,?,?,?,?,?,?);
        '''

        # store per-attack perf for each learner
        for learner, per_attack in self.learner_performance_per_attack.items():
            for attack_class, perf in per_attack.items():
                db.execute(sql, (
                    str(uuid4()),
                    perf.support,
                    perf.f1_score,
                    perf.precision,
                    perf.recall,
                    learner,
                    str(attack_class),
                    self.id,
                ))
                print(learner, attack_class, json.dumps(asdict(perf), indent=4))

        for learner, conf_mat in self.confusion_matrices.items():
            db.execute(r'''
            INSERT INTO ConfusionMatrix(
                id,
                run_id,
                base_learner_name,
                detection_model_name,
                confusion_matrix_image
            )
            VALUES(?, ?,?,?,?);
            ''', (
                       str(uuid4()),
                       self.id,
                       learner,
                       self.detection_model_name,
                       conf_mat
                       ))


        db.commit()

    # # given the run 
    # def retrieve_from_id() -> Run|None:


def fig_to_base64(figure): 
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
