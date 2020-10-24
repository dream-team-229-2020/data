#####
###
### data processing script for cs229 project, 2020
### group: Coleman Smith, Jordan Schuster, Ethan Horoschak
###
#####

DPI = 96  ## for my monitor

import os
import glob
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as fplt


def pickle_data():
    for filename in glob.glob('raw/*.Last.txt'):
        data = pd.read_csv(filename,
                           sep=';',
                           names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                           dtype={'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64, 'volume': np.int64},
                           skip_blank_lines=True,
                           parse_dates=['datetime'],
                           date_parser=lambda date: datetime.strptime(date[:-2], '%Y%m%d %H%M'))

        data.to_pickle('pickles/' + filename[filename.find('\\')+1:] + '.pkl')


# 5, 10, 30 min, 1, 3, 6 hour, 1, 5, 10 day
scales = [5, 10, 30, 60, 180, 360, 1440, 7200, 14400]
def scale_data():
    for picklename in glob.glob('pickles/*'):
        slash_index = picklename.find('\\')
        data = pd.read_pickle(picklename)
        data.to_pickle('processed_pickles/00001 --- ' + picklename[slash_index+1:])
        data.reset_index(inplace=True)
        for scale in scales:
            data['dummy'] = data.index.to_series().apply(lambda n: n // scale)
            data_group= data.groupby('dummy')

            new_data = data_group.sum()
            new_data['datetime'] = data_group.first()['datetime']
            new_data['open'] = data_group.first()['open']
            new_data['high'] = data_group.max()['high']
            new_data['low'] = data_group.min()['low']
            new_data['close'] = data_group.last()['close']
            new_data['volume'] = data_group.sum()['volume']

            new_data.set_index('index', drop=True, inplace=True)
            new_data.reset_index(drop=True, inplace=True)
            data.drop(data.tail(1).index, inplace=True)

            new_data.to_pickle('processed_pickles/' + str(scale).zfill(5) + ' --- ' + picklename[slash_index+1:])

# scale_data()


colors = fplt.make_marketcolors(up='#00ff00', down='#ff0000',
                                edge={'up': '#00ff00', 'down': '#ff0000'},
                                wick={'up': '#00ff00', 'down': '#ff0000'},
                                volume='#0000ff',
                                alpha=1.0)
style = fplt.make_mpf_style(marketcolors=colors, facecolor='#000000', rc={'savefig.facecolor': '#000000'})
def create_save_graph(data, filename):
    data = data.set_index('datetime')
    fplt.plot(data,
              type='candle',
              volume=True,
              figratio=(10, 10),
              show_nontrading=False,
              axisoff=True,
              figsize=(150/DPI, 150/DPI),
              savefig={'fname': filename, 'pad_inches': 0, 'dpi': DPI, 'bbox_inches': 'tight'},
              style=style)



def verify_continuous(data, start, length):
    if start + length >= len(data): return False
    return True
    gap = data.loc[start+1, 'datetime'] - data.loc[start, 'datetime']
    return data.loc[start+length, 'datetime'] - data.loc[start, 'datetime'] == gap * length


sign = lambda x: 1.0 if x > 0 else (-1 if x < 0 else 0)
def get_output(data):
    closes = (data['close']).reset_index(drop=True)
    return (closes[len(closes)-1] - closes[0]) / closes[0]


scales = ['00001', '00005', '00010', '00030', '00060', '00180', '00360', '01440']
def create_dataset():
    for scale in scales[-2:]:
        os.mkdir('dataset/' + scale)
        for picklename in glob.glob('processed_pickles/' + scale + '*.pkl'):
            subdir = 'dataset/' + picklename[:picklename.find(' --- ')][-5:] + '/' + picklename[picklename.find(' --- ')+5: picklename.find('.')] + '/'
            os.mkdir(subdir)
            data = pd.read_pickle(picklename)

            percentages = []
            labels = []

            images_generated = 0
            while images_generated < 200:
                start = random.randrange(len(data))
                if verify_continuous(data, start, 44):
                    create_save_graph(data.loc[start : start+29], subdir + str(images_generated)+'.png')
                    percentages.append(get_output(data.loc[start+29 : start+44]))
                    labels.append(sign(percentages[-1]))
                    images_generated-=-1

            np.savetxt(subdir + 'percentages.txt', np.float64(percentages))
            np.savetxt(subdir + 'labels.txt', np.array(labels))

# os.mkdir('dataset')
create_dataset()


# 150 images from each file per bar type, gives 1800 images per class
