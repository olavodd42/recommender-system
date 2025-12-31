# Usamos a imagem base da NVIDIA (PyTorch) para máxima performance em GPU
# Isso garante compatibilidade total com CUDA drivers
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Metadados
LABEL maintainer="SysRec Architecture Team"
LABEL project="RecSys Masterplan"

# Evitar criação de arquivos .pyc e buffer de stdout
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Diretório de trabalho
WORKDIR /app

# Instalação de dependências do Sistema Operacional (se necessário)
RUN apt-get update && apt-get install -y \
    git \
    htop \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiamos o requirements.txt (Definido no próximo passo)
COPY requirements.txt .

# Instalação das libs Python
# Dica de Arquiteto: Cache do pip para acelerar builds futuros
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expõe portas para Jupyter Lab (8888) e API (8000)
EXPOSE 8888 8000

# Comando padrão: Jupyter Lab (para exploração inicial)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token='recsys'"]