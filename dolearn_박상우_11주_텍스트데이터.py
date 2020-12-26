import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import pandas as pd
import numpy as np


df = pd.DataFrame(columns=('name', 'description', 'category'))

headers = {'Ocp-Apim-Subscription-Key': 'd6f71d9fbf70459baaa2d39cba133a37'}
params = urllib.parse.urlencode({
    'Category':'World',
    'Market':'en-GB',
    'Count':100
})

try:
    conn = http.client.HTTPConnection('api.cognitive.mircosoft.com')
    conn.request('GET','/bing/v7.0/news/?%s' %params, '{body}', headers=headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print(f'[Errno{e.errno}]{e.strerror}')