
def manipulate_listdatasets(dim):
  fp = open("/content/DAIN/datasets/listdatasets.py")
  for i, line in enumerate(fp):
    if i==28:
      number = line[35:39]
      try:
          current_number =int(number)
      except:
          current_number =int(number[:-1])
  fp.close()

  #read input file
  fin = open("/content/DAIN/datasets/listdatasets.py", "rt")
  #read file contents to string
  data = fin.read()
  #replace all occurrences of the required string
  data = data.replace(str(current_number),str(dim))
  #close the input file
  fin.close()
  #open the input file in write mode
  fin = open("/content/DAIN/datasets/listdatasets.py", "wt")
  #overrite the input file with the resulting data
  fin.write(data)
  #close the file
  fin.close()