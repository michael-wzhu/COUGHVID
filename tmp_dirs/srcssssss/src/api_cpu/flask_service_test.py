import json
import time

import urllib.request

import sys
sys.path.append("/")


def test_service(filename, mode):
    header = {'Content-Type': 'application/json'}

    data = {
          "filename": filename,
          "mode": mode,
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
    # filename = "./example_videos/e485460c-63ca-4802-8241-f3469d6ab635.mp3"  # 有问题的咳嗽音
    # filename = "./example_videos/4201-3-1-0.wav"  # 有问题的咳嗽音
    # filename = "./example_videos/4201-3-2-0.wav"  # 有问题的咳嗽音
    # filename = "./example_videos/4201-3-3-0.wav"  # 有问题的咳嗽音
    filename = "./example_videos/0862d8d3-da41-4b96-9b5b-7d0de358f247.mp3"  # 有问题的咳嗽音
    # filename = "./example_videos/c10f7d87-82a2-43c9-a9ba-bfc4bc86b74b.mp3"  # 正常的咳嗽音
    for mode in ["detect_cough", "cough_classfication", "cough_detect_and_classfication"]:

        t0 = time.time()
        test_service(filename, mode)
        t1 = time.time()
        print("time cost: ", t1 - t0)
