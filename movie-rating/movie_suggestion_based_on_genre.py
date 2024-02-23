import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

movie_df = pd.read_csv('movies.csv',
                       names=['Id', 'Title', 'Genres'],
                       header=0,
                       index_col='Id'
                       )
