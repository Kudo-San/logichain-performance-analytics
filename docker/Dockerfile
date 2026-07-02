# 1. Define a imagem base
FROM python:3.9-slim

# 2. Cria o diretório de trabalho dentro do container
WORKDIR /app

# 3. Copia APENAS o arquivo de dependências para o container
# Como nosso contexto no compose é a raiz (.), apontamos para scripts/requirements.txt
COPY scripts/requirements.txt /app/requirements.txt

# 4. Instala as dependências (Aproveita o cache do Docker se o arquivo não mudar)
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5. Copia o restante dos scripts para dentro da imagem
COPY scripts/ /app/scripts/

# 6. Define o comando de inicialização apontando para o loader
CMD ["python", "/app/scripts/data_loader.py"]