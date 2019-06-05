
from data_loading.data_grid import DataGrid




path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)



print(dataGrid.coord(5))
print("(10, 1)")
print(dataGrid.coord(140))
print("(2, 12)")
print(dataGrid.coord(126))
print("(15, 10)")
print(dataGrid.coord(6))
print("(4, 2)")


print(dataGrid.grid_num(10,1))
print("5")
print(dataGrid.grid_num(2,12))
print("140")
print(dataGrid.grid_num(15,10))
print("126")
print(dataGrid.grid_num(4,2))
print("6")

print(dataGrid.data_at(2,2))
print(dataGrid.data_at(2,3))
print(dataGrid.data_at(15,15))
print(dataGrid.data_at(15,14))
print(dataGrid.data_at(8,8)) # only this one is in the grid
