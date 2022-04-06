# ADVANCED PIVOT TABLE
# ON THE ROWS WE HAVE PRIMARY KEY
# THE COLUMN SHOWS THE MOST FREQUENT OCCURRENCIES AMONG CATEGORICAL VARIABLES

import pandas as pd
dataset = pd.read_csv('dataset.csv', sep = ';', encoding='utf-8')

def max_pivottable(DATASET, ROW, COLUMN):
  pivot = pd.pivot_table(DATASET,
                         index = [ROW],
                         columns = [COLUMN],
                         aggfunc = 'size',
                         fill_value = 0)

  pivot_flat = pd.DataFrame(pivot.to_records())
  pivot_flat.set_index(ROW, inplace = True)
  pivot_flat.head()

  pivot_mostfrequent = pd.concat([pivot_flat.index.to_series(),
                                pivot_flat.idxmax(axis=1),
                                pivot_flat.max(axis=1)],
                                 axis = 1).reset_index(drop = True).iloc[1:,:2]
  pivot_mostfrequent.rename(columns = {0:'most_frequent'}, inplace = True)

  pivot_mostfrequent.to_csv('most_frequent.csv')

  pivot_mostfrequent
