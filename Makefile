# 项目特定配置
PROJECT_NAME = researcher

# Namespace 配置（通常不需要修改，所有项目共享）
NAMESPACE_PRODUCTION = ivy
NAMESPACE_TESTING = ivy-testing

# Context 配置
CONTEXT_IVY = ivy      # production/testing 环境使用
CONTEXT_EDGE = edge  # edge 环境使用

K8S_BASE_DIR = .k8s/overlays
PUB_KEY_FILE = pub-key.prod.pem

# 导出变量以供子 Makefile 使用
export PROJECT_NAME
export NAMESPACE_PRODUCTION
export NAMESPACE_TESTING
export CONTEXT_IVY
export CONTEXT_EDGE
export K8S_BASE_DIR
export PUB_KEY_FILE

# 引入通用 Makefile
include scripts/common-makefile/Makefile

# 您可以在这里添加项目特定的其他命令

# 镜像仓库配置
IMAGE_REPO = registry.cn-shanghai.aliyuncs.com/ivysci/gpt-researcher

.PHONY: build custom-command

# 构建并推送镜像 (Target linux/amd64 for server deployment)
build:
	@echo "📦 Building and Pushing image (linux/amd64)..."
	@TAG=$$(git describe --tags --always --dirty); \
	echo "   Tag: $$TAG"; \
	echo ""; \
	docker buildx build --platform linux/amd64 \
		-t $(IMAGE_REPO):$$TAG \
		-t $(IMAGE_REPO):latest \
		--push .; \
	if [ $$? -eq 0 ]; then \
		echo ""; \
		echo "✅ Build & Push complete!"; \
		echo "   Image: $(IMAGE_REPO):$$TAG"; \
		echo ""; \
		echo "📝 To deploy this tag:"; \
		echo "   make set-tag env=testing tag=$$TAG"; \
	else \
		echo ""; \
		echo "❌ Build failed"; \
		exit 1; \
	fi

custom-command:
	@echo "这是项目特定的命令"

# Override seal target to use --from-env-file instead of --from-file
seal:
ifndef env
	@echo "❌ Error: env parameter is required"
	@echo "Usage: make seal env=<environment>"
	@echo "Available environments: $(AVAILABLE_ENVS)"
	@exit 1
endif
	@echo "🔐 生成 $(env) 环境 Sealed Secret (Local Override)..."
	@echo ""

	@# 验证环境参数
	@VALID_ENV="false"; \
	for e in $(AVAILABLE_ENVS); do \
		if [ "$(env)" = "$$e" ]; then \
			VALID_ENV="true"; \
			break; \
		fi; \
	done; \
	if [ "$$VALID_ENV" = "false" ]; then \
		echo "❌ Error: Unknown environment '$(env)'"; \
		echo "Available environments: $(AVAILABLE_ENVS)"; \
		exit 1; \
	fi; \
	\
	case "$(env)" in \
		production) \
			NAMESPACE="$(NAMESPACE_PRODUCTION)"; \
			OVERLAY="production"; \
			;; \
		testing) \
			NAMESPACE="$(NAMESPACE_TESTING)"; \
			OVERLAY="testing"; \
			;; \
		edge-production) \
			NAMESPACE="$(NAMESPACE_PRODUCTION)"; \
			OVERLAY="edge-production"; \
			;; \
		edge-testing) \
			NAMESPACE="$(NAMESPACE_TESTING)"; \
			OVERLAY="edge-testing"; \
			;; \
	esac; \
	\
	OVERLAY_DIR="$(K8S_BASE_DIR)/$$OVERLAY"; \
	SETTINGS_FILE="$$OVERLAY_DIR/settings.yaml"; \
	PUB_KEY="$$OVERLAY_DIR/$(PUB_KEY_FILE)"; \
	SEALED_FILE="$$OVERLAY_DIR/sealed-settings.yaml"; \
	\
	if [ ! -f "$$SETTINGS_FILE" ]; then \
		echo "❌ Error: 配置文件不存在: $$SETTINGS_FILE"; \
		exit 1; \
	fi; \
	\
	if [ ! -f "$$PUB_KEY" ]; then \
		echo "❌ Error: 公钥文件不存在: $$PUB_KEY"; \
		echo "请确保 $(PUB_KEY_FILE) 存在于 $$OVERLAY_DIR"; \
		exit 1; \
	fi; \
	\
	echo "📄 配置文件: $$SETTINGS_FILE"; \
	echo "🔑 公钥文件: $$PUB_KEY"; \
	echo "🔒 输出文件: $$SEALED_FILE"; \
	echo "🏷️  Namespace: $$NAMESPACE"; \
	echo ""; \
	\
	cd "$$OVERLAY_DIR" && \
	kubectl create secret generic $(PROJECT_NAME)-settings \
		--from-env-file=settings.yaml \
		--namespace="$$NAMESPACE" \
		--dry-run=client -o yaml | \
	kubeseal \
		--cert $(PUB_KEY_FILE) \
		--format yaml \
		> sealed-settings.yaml; \
	\
	if [ $$? -eq 0 ]; then \
		echo "✅ $(env) 环境 Sealed Secret 已生成: $$SEALED_FILE"; \
		echo ""; \
		echo "📝 下一步:"; \
		echo "   1. 查看生成的文件: cat $$SEALED_FILE"; \
		echo "   2. 提交到 Git（安全）: git add $$SEALED_FILE"; \
		echo "   3. 部署: make deploy env=$(env)"; \
	else \
		echo ""; \
		echo "❌ 生成 Sealed Secret 失败！"; \
		echo ""; \
		echo "请检查:"; \
		echo "   1. kubeseal 是否已安装: kubeseal --version"; \
		echo "   2. 公钥文件是否存在: $$PUB_KEY"; \
		exit 1; \
	fi
