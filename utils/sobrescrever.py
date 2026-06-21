import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(URL, KEY)

NOME_DO_BUCKET = "csv-mestre"
NOME_ARQUIVO_MESTRE = "dataset_youtube_processado_modulo2.csv"

def sobrescrever_banco():
    print(f"Pegando o '{NOME_ARQUIVO_MESTRE}' local e forçando upload...")
    
    try:
        with open(NOME_ARQUIVO_MESTRE, 'rb') as f:
            supabase.storage.from_(NOME_DO_BUCKET).upload(
                file=f,
                path=NOME_ARQUIVO_MESTRE,
                file_options={
                    "content-type": "text/csv",
                    "x-upsert": "true" # Garante a substituição brutal
                }
            )
        print("Feito. O Supabase foi sobrescrito com o seu arquivo original.")
    except Exception as e:
        print(f"Erro ao sobrescrever: {e}")

if __name__ == "__main__":
    sobrescrever_banco()