# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements et le code de l'application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Exposer le port de l'application
EXPOSE 8000

# Commande pour démarrer l'application avec Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
