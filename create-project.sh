#!/bin/bash

# Content: create_project.sh

# Cria a estrutura de diretórios
mkdir -p {files,documents,config,notebooks,logs,tests}
# Cria arquivos vazios ou com conteúdo inicial
touch apps/{__init__.py,solution.py,utils.py}
touch files/.gitkeep
touch documents/.gitkeep
touch config/config.yaml
touch logs/{app.log,.gitkeep}
touch tests/{__init__.py,test_solution.py}
touch requirements.txt
touch README.md
touch main.py

# Cria o .gitignore usando o serviço da Toptal
curl -sSfL "https://www.toptal.com/developers/gitignore/api/python,pythonvanilla,visualstudiocode,pycharm+all,jupyternotebooks" -o .gitignore