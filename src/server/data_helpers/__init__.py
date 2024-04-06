from typing import Any
from dataclasses import dataclass, asdict
import base64
import io
import sqlite3

from pandas.io.formats.style_render import uuid4

@dataclass
class PerfMetric:
    support: float
    f1_score: float
    precision: float
    recall: float

@dataclass
class OverallPerf: 
    macro_avg: float
    weighted_avg: float
    accuracy: float

@dataclass
class Run: 
    id: str
    detection_model_name: str
    run_tag: str
    learner_performance_per_attack: dict[str, list[PerfMetric]]
    learner_configuration: dict[str, dict[str, Any]]
    learner_overalls: dict[str, OverallPerf]
    timestamp: str
    confusion_matrices: dict[str, str]
    dataset: str
    model_performance: PerfMetric

    def store(self, db: sqlite3.Connection): 
        # store the Run
        sql = r'''
        INSERT INTO Run(id, timestamp, run_tag, detection_model_name) VALUES(?, ?, ?, ?);
        '''
        db.execute(sql, (
            str(uuid4()),
            self.timestamp, 
            self.run_tag,
            self.detection_model_name
        ))
        db.commit()


def fig_to_base64(figure): 
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
