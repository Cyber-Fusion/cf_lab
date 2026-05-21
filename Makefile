COMPOSE := docker compose -f docker/compose.yaml

.PHONY: build up start run exec attach down stop xhost

build:
	$(COMPOSE) build

xhost:
	@xhost +local:root >/dev/null 2>&1 || true

up: xhost
	$(COMPOSE) up -d

exec: up
	$(COMPOSE) exec cf-lab bash

down:
	$(COMPOSE) down
