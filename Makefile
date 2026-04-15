COMPOSE := docker compose -f docker/docker-compose.yaml

.PHONY: build up exec down

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

exec:
	$(COMPOSE) exec cf-lab bash

down:
	$(COMPOSE) down
