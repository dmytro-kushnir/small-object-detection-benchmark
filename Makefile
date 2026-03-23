.PHONY: reproduce-baseline reproduce-exp001 reproduce-exp002 reproduce-exp002b reproduce-exp003 reproduce-ants-smoke reproduce-ants-baseline reproduce-ants-full reproduce-ants-expA002b reproduce-ants-expA003 reproduce-ants-expA003-ablation

reproduce-baseline:
	./scripts/run_smoke_test.sh

reproduce-exp001:
	./scripts/run_exp001.sh

reproduce-exp002:
	./scripts/run_exp002.sh

reproduce-exp002b:
	./scripts/run_exp002b.sh

reproduce-exp003:
	./scripts/run_exp003.sh

# Ant dataset: set ANTS_DATASET_ROOT and run ./scripts/run_ants_prepare.sh before train.
reproduce-ants-smoke:
	./scripts/run_ants_expA000_smoke.sh

# Legacy artifact names (ants_expA000/); prefer reproduce-ants-full for new work.
reproduce-ants-baseline:
	./scripts/run_ants_expA000.sh

reproduce-ants-full:
	./scripts/run_ants_expA000_full.sh

reproduce-ants-expA002b:
	./scripts/run_ants_expA002b.sh

reproduce-ants-expA003:
	./scripts/run_ants_expA003.sh

reproduce-ants-expA003-ablation:
	./scripts/run_ants_expA003_sahi_ablation.sh
