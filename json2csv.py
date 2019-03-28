import os
import glob
import json
import numpy as np
import pandas as pd

column_name = ["filename"]
for i in range(100):
	column_name.append("x{}".format(i))
	column_name.append("y{}".format(i))

# hardlimit of 200 point polygon
supermaxi = 0
# building dataframe list
polylist = []
# dict used to determine if img already used
imgused = {}

os.chdir("./contours")
for file in glob.glob("*.json"):
	# load json file
	jsonfile = open(file).read()
	parsed_json = json.loads(jsonfile)

	# loop through each polygon in the json
	maxi = 0
	for i in range(len(parsed_json["annotations"])):
		link = []
		# check if actualy polygon
		if len(parsed_json["annotations"][i]["segmentation"][0]) < 5:
			continue

		# if img already added skip
		if parsed_json["images"][parsed_json["annotations"][i]["image_id"]-1]["file_name"] in imgused:
			continue

		# add file name and polygon
		link.append(parsed_json["images"][parsed_json["annotations"][i]["image_id"]-1]["file_name"])
		for j in range(200):
			if j < len(parsed_json["annotations"][i]["segmentation"][0]):
				link.append(parsed_json["annotations"][i]["segmentation"][0][j])
			else:
				link.append(np.nan)
		if len(link) > 5:
			polylist.append(tuple(link))

		# mark image as added
		imgused[parsed_json["images"][parsed_json["annotations"][i]["image_id"]-1]["file_name"]] = 1

print(len(polylist))
json_df2 = pd.DataFrame(polylist, columns=column_name)
json_df2.sort_values('filename',inplace=True, ascending=True)
json_df2.to_csv('polygon.csv', index=None)
# print(supermaxi)