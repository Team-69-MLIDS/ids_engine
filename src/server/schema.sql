CREATE TABLE IF NOT EXISTS `init`(
    Initialized INT NOT NULL,
    PRIMARY KEY(Initialized)
);

CREATE TABLE IF NOT EXISTS `DetectionModel`(
    `name` TEXT NOT NULL,
    PRIMARY KEY(`name`)
);

-- a base learner is like XGBClassifier()
CREATE TABLE IF NOT EXISTS `BaseLearner`(
    `name` TEXT NOT NULL,
    PRIMARY KEY(`name`)
);

-- This is the relation between DetectionModel and BaseLearner
-- DetectionModel - < LearnsWith > - BaseLearner
-- Example: LCCDE learns with XGBClassifier, LGBMClassifer, and CatBoostClassifer
CREATE TABLE IF NOT EXISTS `LearnsWith`(
    id UUID NOT NULL,
    detection_model_name TEXT NOT NULL, -- The detection model  (LCCDE)
    base_learner_name TEXT NOT NULL,    -- The name of the base learner (XGBClassifier)
    PRIMARY KEY(`id`),
    FOREIGN KEY (`base_learner_name`) REFERENCES `BaseLearner` (`name`),
    FOREIGN KEY (`detection_model_name`) REFERENCES `DetectionModel` (`name`)
);

CREATE TABLE IF NOT EXISTS `AttackClassification`(
    `name` TEXT NOT NULL,
    PRIMARY KEY(`name`)
);

CREATE TABLE IF NOT EXISTS `ConfusionMatrix`(
    id UUID NOT NULL,
    run_id UUID NOT NULL,
    base_learner_name TEXT NOT NULL,
    detection_model_name TEXT NOT NULL,
    confusion_matrix_image TEXT NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (`run_id`) REFERENCES `Run` (`id`),
    FOREIGN KEY (`base_learner_name`) REFERENCES `BaseLearner` (`name`),
    FOREIGN KEY (`detection_model_name`) REFERENCES `DetectionModel` (`name`)
);

CREATE TABLE IF NOT EXISTS `Run`(
    id UUID NOT NULL,
    timestamp TEXT NOT NULL, 
    run_tag TEXT,                       -- the user provided run id
    detection_model_name TEXT NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (`detection_model_name`) REFERENCES `DetectionModel` (`name`)
);

CREATE TABLE IF NOT EXISTS `PerfMetric`(
    id INT NOT NULL,
    support FLOAT,
    f1_score FLOAT,
    `precision` FLOAT,
    accuracy FLOAT,
    recall FLOAT,
    macro_avg FLOAT,
    weighted_avg FLOAT,
    `BaseLearner_name` TEXT NOT NULL,
    `AttackClassification_id` INT NOT NULL,
    run_id UUID NOT NULL,
    PRIMARY KEY(id),
    FOREIGN KEY (`BaseLearner_name`) REFERENCES `BaseLearner` (`name`),
    FOREIGN KEY (`AttackClassification_id`) REFERENCES `AttackClassification` (`name`),
    FOREIGN KEY (`run_id`) REFERENCES `Run` (`id`)
);

CREATE TABLE IF NOT EXISTS `Hyperparameter`(
    id UUID NOT NULL,
    `description` TEXT,
    `name` TEXT,
    base_learner_name TEXT NOT NULL,    -- the learner that this hyperparam can be used with
    datatype_hint TEXT,                 -- a hint to aid in parsing the value. If null, leave as text or use some other hueristic to parse/validate
    optional INT NOT NULL,
    default_value TEXT,
    PRIMARY KEY(id),
    FOREIGN KEY (`base_learner_name`) REFERENCES `BaseLearner` (`name`)
);

-- Holds a record of all the Hyperparameters and the values used to configure a BaseLearner in a Run.
CREATE TABLE IF NOT EXISTS `LearnerConfig`(
    config_id UUID NOT NULL,
    hyperparameter_id UUID NOT NULL,    -- the parameter id
    base_learner_name TEXT NOT NULL,    -- the name of the base learner this config targets
    run_id UUID NOT NULL,               -- the run in which this this config entry was used
    value TEXT,                         -- needs to be parsed back out. This is kind of a pain point in the design 
    datatype_hint TEXT,                 -- a hint to aid in parsing the value. If null, leave as text or use some other hueristic to parse

    PRIMARY KEY(config_id),
    FOREIGN KEY (`hyperparameter_id`) REFERENCES `Hyperparameter` (`id`),
    FOREIGN KEY (`base_learner_name`) REFERENCES `BaseLearner` (`name`)

);


INSERT OR IGNORE INTO AttackClassification(name) VALUES ("BENIGN"), ("Bot"), ("BruteForce"), ("DoS"), ("Infiltration"), ("PortScan"), ("WebAttack");
INSERT OR IGNORE INTO BaseLearner(name) VALUES ("XGBClassifier"),("CatBoostClassifer"),("LGBMClassifer");
INSERT OR IGNORE INTO DetectionModel(name) VALUES ("lccde"), ("mth"), ("treebased");
-- INSERT OR IGNORE INTO Initialized(Initialized) (0);
-- INSERT OR IGNORE INTO datatype(datatype) values ('string'),('number'),('int'),('float');
