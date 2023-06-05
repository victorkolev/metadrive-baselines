docker-build: 
	docker build --network=host . -f Dockerfile -t moto --network=host

run-dev: 
	docker run --gpus all -it --shm-size=2g --net=host -v `pwd`:/moto moto /bin/bash
