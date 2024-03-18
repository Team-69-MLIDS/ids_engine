from flask import Flask, jsonify, request
import sys
import sqlite3

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

def db_init(db: sqlite3.Connection):
    db_init_script = '''
CREATE TABLE IF NOT EXISTS `DetectionAlgorithm`(
  `name` TEXT NOT NULL,
  author_credits TEXT,
  `source` TEXT,
  `MachineLearningAlgorithm_name` TEXT NOT NULL,
  PRIMARY KEY(`name`)
);

CREATE TABLE IF NOT EXISTS `MachineLearningLibrary`(
  `name` TEXT NOT NULL,
  documentation_url TEXT,
  confusion_matrix JSON,
  `version` TEXT,
  `DetectionAlgorithm_name` TEXT NOT NULL,
  `DetectionAlgorithm_id` INT NOT NULL,
  PRIMARY KEY(`name`),
  FOREIGN KEY (`DetectionAlgorithm_name`) REFERENCES `DetectionAlgorithm` (`name`)
);


CREATE TABLE IF NOT EXISTS `AttackClassification`(
  `name` TEXT NOT NULL,
  PRIMARY KEY(`name`)
);

CREATE TABLE IF NOT EXISTS `ClassificationMetrics`(
  id INT NOT NULL,
  support FLOAT,
  f1_score FLOAT,
  `precision` FLOAT,
  accuracy FLOAT,
  recall FLOAT,
  macro_avg FLOAT,
  weighted_avg FLOAT,
  `MachineLearningLibrary_name` TEXT NOT NULL,
  `AttackClassification_id` INT NOT NULL,
  PRIMARY KEY(id),
  FOREIGN KEY (`MachineLearningLibrary_name`) REFERENCES `MachineLearningLibrary` (`name`),
  FOREIGN KEY (`AttackClassification_id`) REFERENCES `AttackClassification` (`name`)
);

CREATE TABLE IF NOT EXISTS datatype(
  datatype TEXT NOT NULL,
  PRIMARY KEY(datatype)
);

CREATE TABLE IF NOT EXISTS `Hyperparameter`(
  id UUID NOT NULL,
  `value` TEXT COMMENT
    'store as text to support numerous datatypes, then if (datatype == INT) return parseInt(value)...',
  `description` TEXT,
  `name` TEXT,
  `MachineLearningLibrary_name` TEXT NOT NULL,
  datatype_id INT NOT NULL,
  PRIMARY KEY(id)
  FOREIGN KEY (`MachineLearningLibrary_name`) REFERENCES `MachineLearningLibrary` (`name`),
  FOREIGN KEY (datatype_id) REFERENCES datatype (datatype)
);


    ''';
    init_cmds = db_init_script.split(";")
    # print(init_cmds, sep='\n')
    
    for cmd in init_cmds:
        db.execute(cmd)


if __name__=='__main__': 
    db = sqlite3.connect('idsdb.db')
    db_init(db)
    app.run(debug=True)

###
# If a run is successful, you will not get this back. Upon recieving this, the run data will be commited into the db. 
# Some of the algorithms take a while to train however, so set the timeout on the request high and show a spinner or something to indicate.
# If run failed, you will get back an error object that describes the error
run_result = {
  run_id: "Test high learning rate",
  timestamp: "2020-07-15 14:30:26.159446",
  per_lib: [ # each lib has its individual performance measured
    {
      name: "XGBoost",
      cfmat: [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
      ],
      f1_score: 1.0,
      precision: 1.0,
      recall: 0.9,
      macro_avg: 0.9,
      weighted_avg: 0.89,
    },
    {
      name: "CatBoost",
      cfmat: [ 
        [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
      ],
      f1_score: 1.0,
      precision: 1.0,
      recall: 0.9,
      macro_avg: 0.9,
      weighted_avg: 0.89,
    },
  ],
  
  # overall performance of the algorithm (eg. LCCDE takes a combination of multiple libraries and selects the leader model for each attack class)
  overall: {
      f1_score: 1.0,
      precision: 1.0,
      recall: 0.9,
      macro_avg: 0.9,
      weighted_avg: 0.89,
  }
}

###