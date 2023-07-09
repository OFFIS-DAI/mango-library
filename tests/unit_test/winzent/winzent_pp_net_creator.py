import pandapower as pp
import xlsxwriter
net = pp.from_json("grid.json", False)
#pp.to_excel(net, "grid_data.xlsx")
#df = pp.topology.
print(net.values())