#!/bin/bash

# Активация виртуальной среды
source venv/bin/activate

# Обновление pip и установка setuptools
pip install --upgrade pip
pip install setuptools wheel

# Установка зависимостей
pip install -r requirements.txt

echo "Виртуальная среда активирована и зависимости установлены"
echo "Для активации среды используйте: source venv/bin/activate" 