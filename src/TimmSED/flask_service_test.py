import json

import urllib.request


def test_service():
    header = {'Content-Type': 'application/json'}

    data = {
          "filename": "./example_videos/0a2c366d-dda1-4f86-9be9-355edbfdead9.mp3"
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
