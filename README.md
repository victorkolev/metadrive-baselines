# Metadrive simulator with offline data

## Setup

```
make docker-build
make run-dev
```
The commands above will build the docker container and run it in interactive mode. From there, one may initiate a training run with the MetaDrive environment and the provided offline data by loading the environments and data as they are in `train.py` and passing them to an algorithm of choice. One should run `train.py` as follows

```
export logdir="logging/path"
export expert_dir="path/to/data/metadrive_npz"

xvfb-run -a -s "-screen 0 128x168x24 -ac +extension GLX +render -noreset" python -Xfaulthandler train.py --configs vision --expert_datadir $expert_dir --logdir $logdir 
```


