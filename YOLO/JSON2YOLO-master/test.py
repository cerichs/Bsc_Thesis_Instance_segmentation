

with open('new_dir\labels\COCO_Training\Training_Rye_Midsummer_Dense_series1_20_08_20_13_46_38.txt') as f:
    lines = f.read()
lines.replace('1.41269e+06','0')
print(lines)