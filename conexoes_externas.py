import os
#import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# --- SETUP INICIAL ---
load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("🚨 Erro Crítico: Credenciais do Supabase ausentes no .env!")

# Conexão com o Supabase
supabase: Client = create_client(URL, KEY)

# Constantes do seu Storage e Arquivos Locais
NOME_DO_BUCKET = "csv-mestre"
NOME_ARQUIVO_MESTRE = "dataset_youtube_processado_modulo2.csv"
NOME_ARQUIVO_FILA = "fila_pendentes.csv"

# --- FUNÇÕES DE COMUNICAÇÃO EXTERNA ---

def obter_nichos_ativos() -> list:
    """
    Busca a lista de nichos que devem ser processados.
    Preparado para a futura API Fastify, mas usando mock temporário.
    """
    print("[Conexões] Buscando nichos ativos...")
    
    # ========================================================
    # 🔌 ESTRUTURA FUTURA PARA A API FASTIFY
    # ========================================================
    # url_api_fastify = "http://localhost:3000/api/nichos/publicos" # Ajuste depois
    # try:
    #     resposta = requests.get(url_api_fastify)
    #     if resposta.status_code == 200:
    #         # Assumindo que a API retorna algo como: {"nichos": ["nicho1", "nicho2"]}
    #         dados = resposta.json()
    #         return dados.get("nichos", [])
    #     else:
    #         print(f"[Conexões] 🚨 Erro na API: Status {resposta.status_code}")
    # except Exception as e:
    #     print(f"[Conexões] 🚨 Falha ao conectar na API Fastify: {e}")
    #     return []
    # ========================================================

    # VETOR MANUAL (REMOVER QUANDO A API ESTIVER PRONTA)
    print("           -> Usando vetor manual (API não conectada).")
    return [
        "tech review shorts",
    ]

def baixar_csv_mestre() -> pd.DataFrame:
    """
    Faz o download do arquivo mestre no Supabase e salva fisicamente para visualização.
    """
    print(f"\n[Conexões] Baixando '{NOME_ARQUIVO_MESTRE}' do Supabase para o disco...")
    try:
        resposta_bytes = supabase.storage.from_(NOME_DO_BUCKET).download(NOME_ARQUIVO_MESTRE)
        
        # Cria o arquivo físico na pasta
        with open(NOME_ARQUIVO_MESTRE, 'wb') as f:
            f.write(resposta_bytes)
            
        # O Pandas lê o arquivo físico normalmente
        df_mestre = pd.read_csv(NOME_ARQUIVO_MESTRE)
        
        print(f"[Conexões] ✓ CSV salvo e carregado! ({len(df_mestre)} linhas prontas).")
        return df_mestre
        
    except Exception as e:
        print(f"[Conexões] 🚨 Erro ao baixar o CSV mestre: {e}")
        return pd.DataFrame()

def atualizar_csv_mestre(df_atualizado: pd.DataFrame):
    """
    Salva as atualizações no arquivo físico local e faz o upload para o Supabase.
    """
    print(f"\n[Conexões] Salvando arquivo físico e subindo para o Supabase ({len(df_atualizado)} linhas)...")
    try:
        # 1. Atualiza o arquivo físico no PC
        df_atualizado.to_csv(NOME_ARQUIVO_MESTRE, index=False)
        
        # 2. Sobe o arquivo recém-salvo pro Supabase
        with open(NOME_ARQUIVO_MESTRE, 'rb') as f:
            supabase.storage.from_(NOME_DO_BUCKET).upload(
                file=f,
                path=NOME_ARQUIVO_MESTRE,
                file_options={
                    "content-type": "text/csv",
                    "x-upsert": "true"
                }
            )
        print("[Conexões] ✓ Upload concluído! A nova versão já está segura no banco.")
    except Exception as e:
        print(f"[Conexões] 🚨 Erro ao subir o CSV atualizado: {e}")

def limpar_rastro_local():
    """
    Deleta os arquivos CSV da pasta (O Mestre e a Fila). 
    Garante que o diretório inicie limpo na próxima execução.
    """
    print("\n[Conexões] Iniciando varredura de arquivos locais...")
    
    # 1. Limpa o Mestre
    if os.path.exists(NOME_ARQUIVO_MESTRE):
        os.remove(NOME_ARQUIVO_MESTRE)
        print(f"  🧹 Limpeza: O arquivo '{NOME_ARQUIVO_MESTRE}' foi apagado.")
    else:
        print(f"  🧹 Limpeza: Nenhum arquivo '{NOME_ARQUIVO_MESTRE}' encontrado para apagar.")

    # 2. Limpa a Fila
    if os.path.exists(NOME_ARQUIVO_FILA):
        os.remove(NOME_ARQUIVO_FILA)
        print(f"  🧹 Limpeza: O arquivo '{NOME_ARQUIVO_FILA}' foi apagado.")
    else:
        print(f"  🧹 Limpeza: Nenhum arquivo '{NOME_ARQUIVO_FILA}' encontrado para apagar.")

# --- TESTE ISOLADO DO MÓDULO ---
if __name__ == "__main__":
    print("--- Testando módulo conexoes_externas.py ---")
    
    nichos = obter_nichos_ativos()
    print(f"Nichos mapeados: {nichos}")
    
    # Testa baixar, ler e apagar
    df_teste = baixar_csv_mestre()
    if not df_teste.empty:
        print(f"\nColunas detectadas: {df_teste.columns.tolist()[:5]}...")
    
    # Testa a vassoura
    limpar_rastro_local()