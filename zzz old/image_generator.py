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
scales = [5, 10, 30, 60, 180, 360, 1440]
def scale_data():
    os.mkdir('processed_pickles')
    for picklename in glob.glob('pickles/*'):
        slash_index = picklename.find('\\')
        data = pd.read_pickle(picklename)
        data.to_pickle('processed_pickles/0001 --- ' + picklename[slash_index+1:])
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

            new_data.to_pickle('processed_pickles/' + str(scale).zfill(4) + ' --- ' + picklename[slash_index+1:])



colors = fplt.make_marketcolors(up='#000000', down='#ffffff',
                                edge={'up': '#000000', 'down': '#ffffff'},
                                wick={'up': '#000000', 'down': '#ffffff'},
                                volume='#000000',
                                alpha=1.0)
def create_save_graph(data, filename, antialiased):
    data = data.set_index('datetime')
    style = fplt.make_mpf_style(marketcolors=colors, facecolor='#000000', rc={'savefig.facecolor': '#000000', 'lines.antialiased': antialiased, 'patch.antialiased': antialiased})
    fplt.plot(data,
              type='candle',
              # volume=True,
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


sign = lambda x: 1.0 if x >= 0 else 0.0
def get_output(data):
    closes = (data['close']).reset_index(drop=True)
    return (closes[len(closes)-1] - closes[0]) / closes[0]


# scales = ['0001', '0005', '0010', '0030', '0060', '0180', '0360', '1440']
def create_dataset():
    os.mkdir('dataset')
    os.mkdir('dataset/class-0')
    os.mkdir('dataset/class-1')

    os.mkdir('antialiased_dataset')
    os.mkdir('antialiased_dataset/class-0')
    os.mkdir('antialiased_dataset/class-1')

    percentages = []

    total_images_generated = 0
    for picklename in glob.glob('processed_pickles/*.pkl'):
        data = pd.read_pickle(picklename)

        print(picklename)

        num_images_generated = 0
        while num_images_generated < 200:
            start = random.randrange(len(data))
            if verify_continuous(data, start, 44):
                percentages.append(get_output(data.loc[start+29 : start+44]))
                label = int(sign(percentages[-1]))

                create_save_graph(data.loc[start : start+29], 'antialiased_dataset/class-' + str(label) + '/' + str(total_images_generated) + '.png', True)
                create_save_graph(data.loc[start : start+29], 'dataset/class-' + str(label) + '/' + str(total_images_generated) + '.png', False)

                num_images_generated-=-1
                total_images_generated-=-1

    np.savetxt('percentages.txt', np.float64(percentages))


# scale_data()
create_dataset()


# 150 images from each file per bar type, gives 1800 images per class
