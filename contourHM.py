import json
import math

oimg_h = 1600
oimg_w = 1200
nimg_h = 128
nimg_w = 128

json_file = open("wing1.json").read()

parsed_json = json.loads(json_file)
list = parsed_json["annotations"][1]["segmentation"][0]

polygon = []
for i in range(0,len(list), 2):
    point = [list[i], list[i+1]]
    polygon.append(point)


def toSmall(polygon):
    for k in range(len(polygon)):
        polygon[k][0] = int(polygon[k][0]/oimg_h * nimg_h)
        polygon[k][1] = int(polygon[k][1]/oimg_w * nimg_w)

    return polygon

polygon = toSmall(polygon)

print(polygon)

contourList = []
# for i in range(0, len(polygon)):
for i in range(0, 1):
    j = i + 1

    if (i == len(polygon) - 1):
        j = 0

    slope = (polygon[j][1] - polygon[i][1]) / (max(polygon[j][0] - polygon[i][0], 0.000000000001))
    b = - slope * polygon[i][0] + polygon[i][1]

    contourList.append(polygon[i])

    for k in range(min(polygon[i][0], polygon[j][0] + 1), max(polygon[i][0], polygon[j][0] + 1)):
        end = int(slope * k + b)
        point = [k, end]
        contourList.append(point)

print(contourList)
