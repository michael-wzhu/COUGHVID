import json

import urllib.request


def test_service():
    header = {'Content-Type': 'application/json'}

    data = {
          "filename": "./datasets/covid19-cough/raw/00f16a68-4c90-41bc-8a29-3ef2dd1a3ecf.mp3"
        }
    request = urllib.request.Request(
        url='http://127.0.0.1:9000/cough_predict',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )
    response = urllib.request.urlopen(request)
    res = response.read().decode('utf-8')
    result = json.loads(res)
    print(result)


if __name__ == "__main__":
    test_service()
