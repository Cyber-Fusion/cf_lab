COMPOSE := docker compose -f docker/docker-compose.yaml

.PHONY: up exec down

up:
	$(COMPOSE) up -d

exec:
	$(COMPOSE) exec cf-lab bash

down:
	$(COMPOSE) down
