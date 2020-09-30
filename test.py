import re

url = "http://192.168.10.54/zhaobl01/hotupdate-web.git"
re_url = re.findall(r'\S+//\S+/\S+/(\S+).git', url)
print(re_url)