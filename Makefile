.PHONY: start stop restart logs status

start:
	./scripts/dev.sh start

stop:
	./scripts/dev.sh stop

restart:
	./scripts/dev.sh restart

logs:
	./scripts/dev.sh logs

status:
	./scripts/dev.sh status
