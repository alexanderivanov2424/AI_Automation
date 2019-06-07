
import pandas as pd
import numpy as np
import os
import re

class DataGrid:

    def __init__(self, path, regex):
        files = os.listdir(path)


        #regex to parse grid location from file
        pattern = re.compile(regex, re.VERBOSE)

        #load csv files into dictionary
        self.data ={}
        for file in files:
            match = pattern.match(file)
            if(match == None):
                continue
            num = int(match.group("num"))
            self.data[num] = np.array(pd.read_csv(path + file,header=None))

        self.size = len(self.data.keys())
        self.data_length = len(self.data[list(self.data.keys())[0]])
        self.dims = (15,15)

        self.row_sums = [5, 14, 25, 38, 51, 66, 81, 96, 111, 126, 139, 152, 163, 172, 177]
        self.row_starts = [1] + [x + 1 for x in self.row_sums[:-1]]
        self.row_lengths = [5,9,11,13,13,15,15,15,15,15,13,13,11,9,5]
        self.base_vals = [52,26,15,6,7,1,2,3,4,5,13,14,25,38,66]
        self.top_vals = [112,140,153,164,165,173,174,175,176,177,171,172,163,152,126]

    def get_row(self,d):
        return next(i for i,s in enumerate(self.row_sums) if d-s <= 0) + 1
    def up_shift(self,d):
        return (self.row_lengths[self.get_row(d)] + self.row_lengths[self.get_row(d)-1])//2
    def down_shift(self,d):
        return (self.row_lengths[self.get_row(d)-2] + self.row_lengths[self.get_row(d)-1])//2

    def neighbors(self,d):
        neighbor_dict = {}
        if d not in self.row_starts: #left neighbor
            neighbor_dict['left'] = d-1
        if d not in self.row_sums:   #right neighbor
            neighbor_dict['right'] = d+1
        if d not in self.top_vals: #up neighbor
            neighbor_dict['up'] = d + self.up_shift(d)
        if d not in self.base_vals: #down neighbor
            neighbor_dict['down'] = d - self.down_shift(d)
        return neighbor_dict

    # function to get grid location from the grid location number
    def coord(self,d):
        y = self.get_row(d)
        pos_in_row = d
        if y > 1:
            pos_in_row = d - self.row_sums[y-2]
        x = 8 - (self.row_lengths[y-1]+1)//2 + pos_in_row
        return x,y

    #function to get grid location number from coordinate
    def grid_num(self,x,y):
        pos_in_row = x + (self.row_lengths[y-1]+1)//2 - 8
        if y == 1:
            return pos_in_row
        return pos_in_row + self.row_sums[y-2]

    def data_at(self,x,y):
        if not self.in_grid(x,y):
            return None
        return self.data[self.grid_num(x,y)]

    def data_at_loc(self,d):
        x,y = self.coord(d)
        return self.data_at(x,y)

    def get_data_array(self):
        data = np.empty(shape=(self.size,self.data_length))
        for i in range(self.size):
            data[i] = self.data[i+1][:,1]
        return data

    def in_grid(self,x,y):
        if x > 15 or x <= 0:
            return False
        if y > 15 or y <= 0:
            return False
        if y == 1 or y == 15:
            if x < 6 or x > 10:
                return False
        if y == 2 or y == 14:
            if x < 4 or x > 12:
                return False
        if y == 3 or y == 13:
            if x < 3 or x > 13:
                return False
        if y in [4,5,11,12]:
            if x < 2 or x > 14:
                return False
        return True
