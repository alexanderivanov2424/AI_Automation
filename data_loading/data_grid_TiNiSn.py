"""
Example of how to implement data_grid object for specifc data set.

Note:

Regex needs to be correctly formated or else no data will be found.
(this often results in an index out of bounds error)

the "(?P<num>.*?)" is intended to replace the identifying number in the filename.

For example file of the form

params1test.csv
params2test.csv
      3
...

would use the Regex
    params(?P<num>.*?)test.csv

This number will be used as the key in the dataGrid dictionary.    
"""

from data_loading.data_grid import DataGrid



'''
Object to load data from TiNiSn_500C dataset
'''
class DataGrid_TiNiSn_500C(DataGrid):

    def __init__(self,range=None):
        DataGrid.__init__(self,
                          "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/",
                          """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv""",range=None)




'''
Object to load data from TiNiSn_600C dataset
'''
class DataGrid_TiNiSn_600C(DataGrid):

    def __init__(self,range=None):
        DataGrid.__init__(self,
                          "/home/sasha/Desktop/TiNiSn_600C-20190607T173525Z-001/TiNiSn_600C/",
                          """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv""",range=None)
