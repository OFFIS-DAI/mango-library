import pandapower as pp
net = pp.from_json("grid.json", False)
print(net)