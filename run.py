# Helper script to iteratively run the model on each of the 10 test deciles
import json
import time

import main

with open("configs/SU_params.json") as f:
    config = json.load(f)

for i in range(1, 11):

    config["dataset"]["pickle"] = "".join(filter(lambda x: not x.isdigit(), config["dataset"]["pickle"])) + str(i)

    json_object = json.dumps(config, indent=4)

    with open("configs/SU_params.json", "w") as outfile:
        outfile.write(json_object)

    main.main()
    time.sleep(5)
