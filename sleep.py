import requests
import os
import urllib3
import time
urllib3.disable_warnings()
import sys
def download(url):
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:68.0) Gecko/20100101 Firefox/68.0"
  }
  r = requests.get(url=url, headers=headers,verify=False)
  return r.status_code
while download('https://xqy2006.github.io/musicgen/music/'+sys.argv[1]+'.wav')==404:
  print("Page undeploy")
  time.sleep(5)
print("finish")
