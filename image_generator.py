#####
###
### data processing script for cs229 project, 2020
### group: Coleman Smith, Jordan Schuster, Ethan Horoschak
###
#####

import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mplfinance import candlestick_ohlc
import mplfinance as fplt
import matplotlib.dates as mpl_dates
DPI = 96  ## for my monitor

# plt.style.use('ggplot')  # optional
# data['Date'] = pd.to_datetime(data['Date'])
# data['Date'] = data['Date'].apply(mpl_dates.date2num)
# ohlc = ohlc.astype(float)

## filenames come in the form yyyyMMddhhmmyyyyMMddhhmm_MINUTESPERBAR.bmp
colors = fplt.make_marketcolors(up='#00ff00', down='#ff0000',
                                edge={'up': '#00ff00', 'down': '#ff0000'},
                                wick={'up': '#00ff00', 'down': '#ff0000'},
                                volume='#0000ff',
                                alpha=1.0)
style = fplt.make_mpf_style(marketcolors=colors, facecolor='#000000', rc={'savefig.facecolor': '#000000'})
def create_save_graph(data, filename):
    fplt.plot(data,
              type='candle',
              volume=True,
              figratio=(10, 10),
              show_nontrading=False,
              axisoff=True,
              figsize=(150/DPI, 150/DPI),
              savefig={'fname': filename, 'pad_inches': 0, 'dpi': DPI, 'bbox_inches': 'tight'},
              style=style)



for filename in glob.glob('*.Last.txt'):
    data = pd.read_csv(filename,
                       sep=';',
                       names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                       index_col='datetime',
                       dtype={'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64, 'volume': np.int64},
                       skip_blank_lines=True,
                       parse_dates=['datetime'],
                       date_parser=lambda date: datetime.strptime(date[:-2], '%Y%m%d %H%M'))

    create_save_graph(data[:30], 'test/'+filename+'.png')

# 100 images from each file per bar type, gives 1200 images per class
