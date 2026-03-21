.PHONY: reproduce-baseline reproduce-exp001 reproduce-exp002

reproduce-baseline:
	./scripts/run_smoke_test.sh

reproduce-exp001:
	./scripts/run_exp001.sh

reproduce-exp002:
	./scripts/run_exp002.sh
