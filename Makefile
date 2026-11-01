# Apollo 11 simulation (consumer) — gates, binding, tests
PY := venv/bin/python
PKG ?= nominal_packages/apollo11_nominal_package.json
SEED ?= 42
FLEET_DIR ?= outputs/fleet_jul16_s$(SEED)
LOCAL_DIR ?= outputs/fleet_local25_s$(SEED)
WORKERS ?= 10
ENV_PREFIX ?= $(APOLLO_ENV_PREFIX)
FLEET_ENV = APOLLO_DELIVERED_PACKAGE=$(PKG) $(ENV_PREFIX)
PINNED = ei_target.json bplane_target.json ca_target.json nominal_results.json od_cov.json run_config.json delivered_package.json

.PHONY: test test-fast check-artifact check-package self-test refly refly-test clean fleet-setup fleet-preflight fleet-smoke fleet-local

test:
	$(PY) -m pytest -q tests/

test-fast:          # gates, binding, launch+TLI re-fly; skips the physics self-test and the full re-fly
	$(PY) -m pytest -q tests/ -m "not refly" --deselect tests/test_check_delivered_nominal.py::test_self_test_every_check_fires

check-artifact:
	$(PY) check_artifact.py $(PKG)

check-package:
	$(PY) check_delivered_nominal.py $(PKG)

self-test:
	$(PY) check_delivered_nominal.py --self-test $(PKG)

refly:              # deterministic re-fly of the delivered nominal -> outputs/refly_report.json
	$(PY) refly_delivered_nominal.py $(PKG)

refly-test:
	$(PY) -m pytest -q tests/test_refly_parity.py

# Definitive config (the original's): make fleet-setup fleet-local ENV_PREFIX="APOLLO_HAZARDS=1 APOLLO_IGM_ROBUST=1 APOLLO_SIVB_RESTART=1 APOLLO_APS_IGNITION=1" \
#     FLEET_DIR=outputs/fleet_jul16_s42_haz LOCAL_DIR=outputs/fleet_local25_s42_haz
fleet-setup:        # sim plan S5: the setup pin (bound nominal + targets + OD covariances + run_config) -> $(FLEET_DIR)
	$(FLEET_ENV) $(PY) cluster_run.py setup $(FLEET_DIR) 250 $(SEED)
	$(FLEET_ENV) $(PY) cluster_run.py preflight $(FLEET_DIR)

fleet-preflight:
	$(FLEET_ENV) $(PY) cluster_run.py preflight $(FLEET_DIR)

fleet-smoke:        # trials 0,1 on 2 spawned workers from a copy of the pin; exit 1 if a worker flew unbound
	$(FLEET_ENV) $(PY) fleet_smoke.py $(FLEET_DIR) outputs/fleet_smoke_s$(SEED) 25 $(SEED) 0,1

fleet-local:        # the local 25 on $(WORKERS) workers from a COPY of the pin (never write shards into the cluster pin)
	mkdir -p $(LOCAL_DIR)
	for f in $(PINNED); do [ -e $(LOCAL_DIR)/$$f ] || cp $(FLEET_DIR)/$$f $(LOCAL_DIR)/; done
	$(FLEET_ENV) $(PY) cluster_run.py preflight $(LOCAL_DIR)
	$(FLEET_ENV) $(PY) cluster_run.py shard $(LOCAL_DIR) 25 $(SEED) 0 1 $(WORKERS)
	$(PY) cluster_run.py merge $(LOCAL_DIR) 1

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
