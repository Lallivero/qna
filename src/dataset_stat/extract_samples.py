import json
import numpy as np

if __name__ == "__main__":
    fileName = "./dataset/json/dev.json"
    with open(fileName, "r") as fp:
        devData = json.load(fp)

    # keyList = devData["data"].keys()

    # print np.array(range(len(keyList) ) )
    devData = devData["data"]

    idx = np.random.choice(np.array(range(len(devData) ) ), 20, replace=False)
    sampleDevData = list()
    for keyId in idx:
        sampleDevData.append(devData[keyId] )

    assert len(sampleDevData) == 20
    sampleFileName = "./dataset/json/sample-dev.json"
    with open(sampleFileName, "w") as fp:
        json.dump(sampleDevData, fp)
