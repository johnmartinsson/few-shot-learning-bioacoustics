def get_class_name_train(label):
    classes = [
        'SNMK',
        'c_4',
        'c_26',
        'c_13',
        'c_22',
        'c_9',
        'c_11',
        'c_8',
        'c_24',
        'GIG',
        'c_21',
        'GRN',
        'COYE',
        'RBGR',
        'CHSP',
        'c_16',
        'c_17',
        'BTBW',
        'c_12',
        'BBWA',
        'c_20',
        'c_19',
        'WHP',
        'c_7',
        'AMRE',
        'WTSP',
        'GCTH',
        'c_23',
        'c_6',
        'c_14',
        'SAVS',
        'RUM',
        'AGGM',
        'c_2',
        'c_25',
        'CALL',
        'SQT',
        'SWTH',
        'SOCM',
        'c_3',
        'CCMK',
        'c_5',
        'c_10',
        'OVEN',
        'c_15',
        'c_1',
        'c_18',
        'UNK'
    ]
    return classes[label]

def get_label_train(row, columns):
    classes = [
        'SNMK',
        'c_4',
        'c_26',
        'c_13',
        'c_22',
        'c_9',
        'c_11',
        'c_8',
        'c_24',
        'GIG',
        'c_21',
        'GRN',
        'COYE',
        'RBGR',
        'CHSP',
        'c_16',
        'c_17',
        'BTBW',
        'c_12',
        'BBWA',
        'c_20',
        'c_19',
        'WHP',
        'c_7',
        'AMRE',
        'WTSP',
        'GCTH',
        'c_23',
        'c_6',
        'c_14',
        'SAVS',
        'RUM',
        'AGGM',
        'c_2',
        'c_25',
        'CALL',
        'SQT',
        'SWTH',
        'SOCM',
        'c_3',
        'CCMK',
        'c_5',
        'c_10',
        'OVEN',
        'c_15',
        'c_1',
        'c_18',
        'UNK'
    ]
    positives = list(columns[3:][row[1][3:] == 'POS'])
    if len(positives) == 0:
        y_class = classes.index('UNK')
    elif len(positives) == 1:
        y_class = classes.index(positives[0])
    else:
        raise ValueError("Unhandled case in get_label")
    
    return y_class

def get_label_valid(row, columns):
    classes = [
        'Q',
        'UNK'
    ]
    positives = list(columns[3:][row[1][3:] == 'POS'])
    if len(positives) == 0:
        y_class = classes.index('UNK')
    elif len(positives) == 1:
        y_class = classes.index(positives[0])
    else:
        raise ValueError("Unhandled case in get_label")
    
    return y_class

def get_class_name_valid(label):
    classes = [
        'Q',
        'UNK'
    ]
    return classes[label]
