from docstring_parser import parse_from_object, parse
from catboost import CatBoostClassifier
import csv
import re
import docstrings
import structlog

#setup logging
structlog.stdlib.recreate_defaults()
log = structlog.get_logger('docgen')

LIGHTGBM_IGNORE = ['**kwargs']
CATBOOST_IGNORE = []
XGBOOST_IGNORE = [
        'interaction_constraints',
        'feature_types',
        'eval_metric',
        'callbacks',
        'kwargs'
        ]

CSV_HEADER: list[str] = ['algorithm', 'parameter_name', 'typeinfo', 'default', 'description', 'optional'  ]

ALGORITHM_NAMES = [
        'CatBoostClassifier',
        'LGBMClassifier',
        'XGBClassifier',
        ]

from sklearn.tree import DecisionTreeClassifier

t = parse_from_object(DecisionTreeClassifier)
for p in t.params:
    print(p.arg_name)

pass
log.debug('Parsing hyperparameters')
with open('hyperparams.csv', mode='w+', newline='') as csvfile: 
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(CSV_HEADER)
    # write catboostclassifier params
    d = parse_from_object(CatBoostClassifier)
    d = [p for p in d.params if p.arg_name not in CATBOOST_IGNORE]
    log.info('Parsing CatBoostClassifier params...')
    for p in d:
        # extract default
        match = re.search("default=(.*)]", p.type_name or 'any')
        if match and p.type_name: 
            default_value = p.type_name[match.start()+8:match.end()-1]
        else:
            default_value = "no default"
        # extract plain typename
        type_name = p.type_name or 'No type'
        first_comma = re.search(r',|\[', type_name)
        if first_comma:
            type_name = type_name[:first_comma.end()-1]
            type_name = re.sub(r'\s+or\s+', ', ', type_name)
        # print(p.arg_name, type_name, default_value)
        # make everything optional
        csv_writer.writerow(('CatBoostClassifier', p.arg_name, type_name, default_value, p.description or "No description", True))
    log.info('Parsed CatBoostClassifier params.')

    # write lightgbm params
    log.info('Parsing LGBMClassifier params...')
    d = parse(docstrings.lightgbm_docs)
    d = [p for p in d.params if p.arg_name not in LIGHTGBM_IGNORE]
    for p in parse(docstrings.lightgbm_docs).params:
        # figure out if its optional or not
        is_optional = re.search('optional', p.type_name or 'any') != None
        # extract default 
        match = re.search(r'default=(.*)', p.type_name or 'any')
        if match and p.type_name:
            default_value = p.type_name[match.start()+8:match.end()-1]
        else:
            default_value = "No default"
        # print(p.arg_name, p.type_name, is_optional, default_value, p.description)
        csv_writer.writerow(('LGBMClassifier', p.arg_name, p.type_name, p.default or "No default", p.description or "No description", is_optional))
    log.info('Parsed LGBMClassifier params.' )

    # write xgboost params
    log.info('Parsing XGBClassifier params...')
    d = parse(docstrings.xgboost_doc)
    d = [p for p in d.params if p.arg_name not in XGBOOST_IGNORE]
    for p in d:
        typestr = p.type_name or "any"
        optional = False
        if 'Optional' in typestr:
            optional = True
            typematch = re.search(r'\[\w+\]', typestr)
            if typematch:
                typestr = typestr[typematch.start()+1:typematch.end()-1]
        csv_writer.writerow(('XGBClassifier', p.arg_name, typestr or 'any', p.default or "No default", p.description or "No description", optional))
    log.info('Parsed XGBClassifier params.')
log.info('Done.')
        

