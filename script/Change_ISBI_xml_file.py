# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:09:49 2021

@author: Martin_Priessner
"""


# Select the ground truth xml in ISBI format
xml_file  =r"F:\Martin G-Drive\1.1_Imperial_College\20210715_Tracking\TH10Qu22Dur14\GT9\MAX_2_ISBI.xml"

# Select the location and name of the new generated  xml file containing just the relevant timepoints
new_xml_file = r"F:\Martin G-Drive\1.1_Imperial_College\20210715_Tracking\TH10Qu22Dur14\GT9\MAX_2_ISBI_DS.xml"

# specify the number of images in the ground truth file
input_slice_nr = 39

# Select the speed (velocity), whether the file was down-sampled (taking every second image) and number for the image limit
velocity = 1
down_sample = True
image_limit = 20


######Running the code below generates the new XML format in ISBI format ready to be used in the ICY Batch Score plugin for tracking quality evaluation
##############################################################################
#create list of all the numbers that should be considerd for tracking
counter = 0
time_point_list = []
while len(time_point_list)<image_limit:
    if down_sample:
        number = counter * (velocity * 2) 
    else:
        number = counter * (velocity) 
    time_point_list.append(number)
    counter +=1

all_number = []
highest_number = time_point_list[-1]

# create a list with all the time point numbers possible for in the xml file
all_number = [i for i in range(1, input_slice_nr+1)]

 # select all the numbers that should be exluded
remove_time_point_list = [i for i in all_number if i not in time_point_list]


# Write new file which removes all the not necessary datapoint
lever = True
false_counter = 0
with open(new_xml_file,'w') as new_file:
    with open(xml_file) as f:
        lines = f.readlines()
        for line in lines:
            if "<detection" in line:
                for rm_time in remove_time_point_list:
                    if 't="%s"'%rm_time in line:
                        lever = False
                if lever == True:
                    # check which number is present in that line
                    for number in time_point_list:
                        if 't="%s"'%number in line:
                            tp_value = number
                    # get the index of that number in the time_point_list 
                    index = time_point_list.index(tp_value)
                    # replace the number in the line for the new number
                    line = line.replace('t="%s"'%tp_value, 't="%s"'%index)  
                    # write the new line in the file
                    new_file.write(line)
                lever = True
            else:
                new_file.write(line)



#### Good luck with your experiments ####     
