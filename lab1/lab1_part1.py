import requests
#Part 1: HTTP Request in Python
url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_string.php'
r = requests.post(url, data = {'netid':'nnathan2', 'name':'Naveen Nathan'})
print(r)

list = []
generator = (num * 498 for num in range(400))
list.append([r.text[num] for num in generator])
message = ''.join(list[0])
print(message)