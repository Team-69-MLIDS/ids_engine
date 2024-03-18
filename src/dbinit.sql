CREATE TABLE IF NOT EXISTS `DetectionAlgorithm`(
  `name` TEXT NOT NULL,
  PRIMARY KEY(`name`)
);

CREATE TABLE IF NOT EXISTS `MachineLearningLibrary`(
  `name` TEXT NOT NULL,
  PRIMARY KEY(`name`)
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
  `description` TEXT,
  `name` TEXT,
  `MachineLearningLibrary_name` TEXT NOT NULL,
  datatype_id INT NOT NULL,
  PRIMARY KEY(id),
  FOREIGN KEY (`MachineLearningLibrary_name`) REFERENCES `MachineLearningLibrary` (`name`),
  FOREIGN KEY (datatype_id) REFERENCES datatype (datatype)
);

INSERT OR IGNORE INTO AttackClassification(name) VALUES ("BENIGN"), ("Bot"), ("BruteForce"), ("DoS"), ("Infiltration"), ("PortScan"), ("WebAttack");
INSERT OR IGNORE INTO MachineLearningLibrary(name) VALUES ("XGBoost"),("CatBoost"),("LightGBM");
INSERT OR IGNORE INTO datatype(datatype) values ('string'),('number'),('int'),('float');