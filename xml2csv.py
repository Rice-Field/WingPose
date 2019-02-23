import os
import glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def xml2np(path, xml_list = []):
	tree = ET.parse(path)
	root = tree.getroot()
	images = root.find('images')
	for member in images.findall('image'):
		# print(member[0][0].text)
		if member[0][0].text == 'bad':
			continue
		print(member.get('file'))
		value = (member.get('file'),
			int(member[0][1].get('x')), int(member[0][1].get('y')),
			int(member[0][2].get('x')), int(member[0][2].get('y')),
			int(member[0][3].get('x')), int(member[0][3].get('y')),
			int(member[0][4].get('x')), int(member[0][4].get('y')),
			int(member[0][5].get('x')), int(member[0][5].get('y')),
			int(member[0][6].get('x')), int(member[0][6].get('y')),
			int(member[0][7].get('x')), int(member[0][7].get('y')),
			int(member[0][8].get('x')), int(member[0][8].get('y')))
		xml_list.append(value)
	column_name = ['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
				 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']
	xml_df = pd.DataFrame(xml_list, columns=column_name)
	return xml_df, xml_list


label_path = 'truewing.xml'
xml_df, xml_list = xml2np(label_path)
label_path = 'truewing2.xml'
xml_df, xml_list = xml2np(label_path, xml_list)

xml_df.to_csv('truewing.csv', index=None)
print('Successfully converted xml to csv.')
print(len(xml_list))

