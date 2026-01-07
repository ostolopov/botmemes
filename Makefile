.PHONY: help install setup export build fetch schedule clean venv deactivate

# Переменные
PYTHON = python3
VENV = venv
VENV_BIN = $(VENV)/bin
PIP = $(VENV_BIN)/pip
PYTHON_VENV = $(VENV_BIN)/python3

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Показать справку по командам
	@echo "$(GREEN)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: venv install ## Полная настройка проекта (venv + зависимости)

venv: ## Создать виртуальное окружение
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(GREEN)Создаю виртуальное окружение...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	else \
		echo "$(YELLOW)Виртуальное окружение уже существует$(NC)"; \
	fi

install: venv ## Установить зависимости
	@echo "$(GREEN)Устанавливаю зависимости...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Зависимости установлены$(NC)"

export: venv ## Экспортировать посты из Telegram-канала
	@echo "$(GREEN)Экспорт постов из канала...$(NC)"
	@$(PYTHON_VENV) main.py export_telegram

export-limit: venv ## Экспортировать ограниченное количество постов (использовать: make export-limit LIMIT=500)
	@if [ -z "$(LIMIT)" ]; then \
		echo "$(YELLOW)Укажите LIMIT: make export-limit LIMIT=500$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Экспорт $(LIMIT) постов из канала...$(NC)"
	@EXPORT_LIMIT=$(LIMIT) $(PYTHON_VENV) main.py export_telegram

build: venv ## Построить эмбеддинги и FAISS-индекс
	@echo "$(GREEN)Построение эмбеддингов и индекса...$(NC)"
	@$(PYTHON_VENV) main.py build_embeddings

fetch-reddit: venv ## Загрузить мемы с Reddit (использовать: make fetch-reddit LIMIT=50)
	@LIMIT=$${LIMIT:-50}; \
	echo "$(GREEN)Загрузка мемов с Reddit (лимит: $$LIMIT)...$(NC)"; \
	$(PYTHON_VENV) main.py fetch_reddit --limit $$LIMIT

fetch-reddit-no-taste: venv ## Загрузить мемы с Reddit без фильтрации по вкусу
	@LIMIT=$${LIMIT:-50}; \
	echo "$(GREEN)Загрузка мемов с Reddit без фильтрации (лимит: $$LIMIT)...$(NC)"; \
	$(PYTHON_VENV) main.py fetch_reddit --limit $$LIMIT --no-taste

schedule: venv ## Отправить мемы в отложку (использовать: make schedule MAX=10 INTERVAL=1 DELAY=5)
	@MAX=$${MAX:-10}; \
	INTERVAL=$${INTERVAL:-1.0}; \
	DELAY=$${DELAY:-5}; \
	echo "$(GREEN)Отправка $$MAX мемов в отложку (интервал: $$INTERVAL ч, задержка: $$DELAY мин)...$(NC)"; \
	$(PYTHON_VENV) main.py schedule_posts --max $$MAX --interval $$INTERVAL --delay $$DELAY

auto: venv ## Автономный режим: автоматический поиск и отправка мемов (использовать: make auto SEARCH_INTERVAL=1 POSTS_PER_SEARCH=50)
	@SEARCH_INTERVAL=$${SEARCH_INTERVAL:-1.0}; \
	POSTS_PER_SEARCH=$${POSTS_PER_SEARCH:-50}; \
	POSTS_TO_SCHEDULE=$${POSTS_TO_SCHEDULE:-10}; \
	SCHEDULE_INTERVAL=$${SCHEDULE_INTERVAL:-2.0}; \
	SCHEDULE_DELAY=$${SCHEDULE_DELAY:-10}; \
	TASTE_THRESHOLD=$${TASTE_THRESHOLD:-0.6}; \
	echo "$(GREEN)Запуск автономного режима...$(NC)"; \
	echo "  Интервал поиска: $$SEARCH_INTERVAL часов"; \
	echo "  Постов за поиск: $$POSTS_PER_SEARCH"; \
	echo "  Мемов в отложку: $$POSTS_TO_SCHEDULE"; \
	echo "  Интервал между постами: $$SCHEDULE_INTERVAL часов"; \
	echo "$(YELLOW)Нажмите Ctrl+C для остановки$(NC)"; \
	$(PYTHON_VENV) main.py auto \
		--search-interval $$SEARCH_INTERVAL \
		--posts-per-search $$POSTS_PER_SEARCH \
		--posts-to-schedule $$POSTS_TO_SCHEDULE \
		--schedule-interval $$SCHEDULE_INTERVAL \
		--schedule-delay $$SCHEDULE_DELAY \
		--taste-threshold $$TASTE_THRESHOLD

full-pipeline: export build fetch-reddit ## Полный пайплайн: экспорт → эмбеддинги → поиск с Reddit
	@echo "$(GREEN)✓ Полный пайплайн выполнен$(NC)"

pipeline-and-schedule: full-pipeline schedule ## Полный пайплайн + отправка в отложку

clean: ## Очистить временные файлы и данные
	@echo "$(YELLOW)Очистка временных файлов...$(NC)"
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf *.pyc */*.pyc
	@rm -rf .pytest_cache
	@echo "$(GREEN)✓ Очистка завершена$(NC)"

clean-data: ## Очистить все данные (медиа, эмбеддинги, кандидаты)
	@echo "$(YELLOW)⚠️  ВНИМАНИЕ: Это удалит все данные!$(NC)"
	@read -p "Продолжить? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/; \
		rm -f *.session *.session-journal; \
		echo "$(GREEN)✓ Данные удалены$(NC)"; \
	else \
		echo "$(YELLOW)Отменено$(NC)"; \
	fi

deactivate: ## Показать команду для деактивации venv
	@echo "$(YELLOW)Для деактивации виртуального окружения выполните:$(NC)"
	@echo "  deactivate"

check-env: ## Проверить наличие .env файла
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  Файл .env не найден!$(NC)"; \
		echo "Создайте файл .env с настройками:"; \
		echo "  TELEGRAM_API_ID=..."; \
		echo "  TELEGRAM_API_HASH=..."; \
		echo "  TELEGRAM_CHANNEL_USERNAME=@..."; \
		exit 1; \
	else \
		echo "$(GREEN)✓ Файл .env найден$(NC)"; \
	fi

check-model: ## Проверить наличие модели вкуса
	@if [ ! -f mean_taste_multimodal.npy ]; then \
		echo "$(YELLOW)⚠️  Файл mean_taste_multimodal.npy не найден!$(NC)"; \
		echo "Модель вкуса не будет использоваться."; \
	else \
		echo "$(GREEN)✓ Модель вкуса найдена$(NC)"; \
	fi

status: check-env check-model ## Проверить статус проекта
	@echo "$(GREEN)Проверка проекта...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "$(GREEN)✓ Виртуальное окружение создано$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Виртуальное окружение не создано (выполните: make setup)$(NC)"; \
	fi
	@if [ -d "data/embeddings" ] && [ -f "data/embeddings/index.faiss" ]; then \
		echo "$(GREEN)✓ FAISS-индекс построен$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  FAISS-индекс не построен (выполните: make build)$(NC)"; \
	fi

.DEFAULT_GOAL := help

