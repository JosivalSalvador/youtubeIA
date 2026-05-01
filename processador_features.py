import pandas as pd
import numpy as np
from datetime import datetime, timezone
import re
import emoji
import ast
import textwrap
from deep_translator import GoogleTranslator
import os
import json
import time
import requests
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

def remover_colunas_inuteis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas de metadados irrelevantes para a análise do modelo.
    Garante limpeza estrutural do DataFrame de forma segura.
    Atualizada com as variáveis descontinuadas do pipeline de emergência.
    """
    # Lista exata fundindo as colunas originais inúteis com as novas colunas
    # mapeadas para remoção na Camada Prata.
    colunas_para_remover = [
        # Lixo estrutural original
        'restricao_idade',
        'live_inicio_agendado',
        'live_fim_agendado',
        'live_inicio_real',
        'live_fim_real',
        'espectadores_simultaneos',
        'data_gravacao',
        'local_gravacao_desc',
        'latitude',
        'longitude',
        'dimensao',
        'projecao',
        
        # Novas colunas removidas pelo pipeline de emergência
        'favoritos',
        'duracao_iso',
        'definicao',
        'privacidade',
        'licenca',
        'permite_embed',
        'estatisticas_publicas',
        'restricao_regiao_bloqueada',
        'restricao_regiao_permitida',
        'topicos_ids',
    ]

    # Realiza o drop das colunas no dataframe principal
    # O errors='ignore' garante que o código não quebre caso uma dessas colunas já não exista
    df = df.drop(columns=colunas_para_remover, errors='ignore')

    print(f" Etapa 1 Concluída: {len(colunas_para_remover)} colunas inúteis ou descontinuadas listadas para remoção.")
    print(f"Total de colunas restantes: {df.shape[1]}")
    
    return df

def tratar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes (NaN) e anomalias de tipagem nas colunas de texto e imagens.
    Garante que as operações futuras não quebrem por falta de dados.
    """
    # 1. Tratamento de Textos (Aplica a blindagem do código de emergência direto na raiz)
    # Garante que as colunas principais não fiquem como NaN nativo nem como a string "nan"
    colunas_texto = ['titulo', 'descricao', 'texto_falado']
    for col in colunas_texto:
        if col in df.columns:
            # Preenche o NaN real do Pandas
            df[col] = df[col].fillna("")
            
            # Blindagem: Mata a anomalia do Pandas que converte nulo para a string "nan"
            mascara_string_nan = df[col].astype(str).str.strip().str.lower() == "nan"
            df.loc[mascara_string_nan, col] = ""

    # 2. Tratamento e Descarte de Thumbnails
    if 'thumb_maxres' in df.columns and 'thumb_default' in df.columns:
        # Salva as imagens faltantes
        df['thumb_maxres'] = df['thumb_maxres'].fillna(df['thumb_default'])
        
        # Como já cumpriu o seu propósito, deletamos a coluna inútil aqui mesmo!
        df = df.drop(columns=['thumb_default'], errors='ignore')

    print(" Etapa 2 Concluída: Nulos tratados, anomalias resolvidas e 'thumb_default' descartada.")
    
    # Check rápido para garantir que zeramos os problemas
    if 'descricao' in df.columns:
        print(f"Quantidade de nulos em 'descricao' agora: {df['descricao'].isna().sum()}")
    if 'thumb_maxres' in df.columns:
        print(f"Quantidade de nulos em 'thumb_maxres' agora: {df['thumb_maxres'].isna().sum()}")
        
    return df

def limpar_erros_transcricao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa mensagens de erro da raspagem na coluna 'texto_falado', 
    substituindo-as por strings vazias para não corromper as features de NLP.
    """
    if 'texto_falado' in df.columns:
        # Cria máscaras booleanas para identificar os erros exatos
        mask_erro_tag = df['texto_falado'].str.startswith('[ERRO_', na=False)
        mask_erro_http = df['texto_falado'].str.contains('httpsconnectionpool', case=False, na=False)

        # Conta quantas linhas serão limpas para termos controle
        linhas_com_erro = (mask_erro_tag | mask_erro_http).sum()

        # Aplica a string vazia nas linhas que deram match
        df.loc[mask_erro_tag | mask_erro_http, 'texto_falado'] = ""

        print(f"Etapa 3 Concluída: {linhas_com_erro} linhas com falhas de raspagem no 'texto_falado' foram esvaziadas.")
    else:
        print("A coluna 'texto_falado' não foi encontrada no DataFrame.")
        
    return df

def criar_features_tracao_metadados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa Engenharia de Tração e Metadados temporais.
    Calcula idade dinâmica, velocidade de views, engajamento e o Score Viral.
    """
    print("Iniciando Etapa 4: Processando Engenharia de Tração e Metadados...")

    try:
        # 1. Metadados Temporais (Dia e Hora)
        df['data_publicacao_dt'] = pd.to_datetime(df['data_publicacao'], utc=True)
        
        # Extraindo o dia da semana já formatado em minúsculo na raiz
        df['dia_postagem'] = df['data_publicacao_dt'].dt.day_name().str.lower()
        df['hora_postagem'] = df['data_publicacao_dt'].dt.hour

        # 2. Segurança de Tipagem (Garante que os dados crus sejam números limpos)
        for col in ['visualizacoes', 'curtidas', 'comentarios']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 3. Engenharia de Tração (Idade Dinâmica e Velocidade Pura)
        data_atual = datetime.now(timezone.utc)
        idade_bruta = (data_atual - df['data_publicacao_dt']).dt.days

        # O teto da idade respeita o limite máximo real do próprio dataset.
        # O piso mínimo é 1 para evitar a quebra matemática (divisão por zero).
        max_idade = idade_bruta.max() if not idade_bruta.empty else 1
        df['idade_dias'] = np.clip(idade_bruta, 1, max_idade)

        df['velocidade_views'] = df['visualizacoes'] / df['idade_dias']

        # 4. Taxas Exatas (Sem mascarar views)
        df['taxa_conversao'] = np.where(
            df['visualizacoes'] > 0,
            (df['curtidas'] / df['visualizacoes']) * 100,
            0
        )
        df['taxa_discussao'] = np.where(
            df['visualizacoes'] > 0,
            (df['comentarios'] / df['visualizacoes']) * 100,
            0
        )

        # 5. O Score Viral Universal (Peso real para os comentários)
        PESO_COMENTARIO = 5.0
        multiplicador_engajamento = 1 + (df['taxa_conversao'] / 100) + ((df['taxa_discussao'] / 100) * PESO_COMENTARIO)
        df['score_viral'] = df['velocidade_views'] * multiplicador_engajamento

        # Limpeza: Deletamos a coluna datetime temporária
        df = df.drop(columns=['data_publicacao_dt'], errors='ignore')

        print(" SUCESSO ABSOLUTO!")
        print(" -> Novas colunas adicionadas:")
        print("    [Tempo]: dia_postagem, hora_postagem, idade_dias")
        print("    [Tração]: velocidade_views, taxa_conversao, taxa_discussao, score_viral")
        print("\nO Reranker agora tem olhos matemáticos. Pode ir para a próxima etapa!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 4: {e}")
        print("Dica: Verifique se a coluna 'data_publicacao' ou as numéricas estão corretas no DataFrame.")

    return df

def criar_features_formato_copywriting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa Engenharia de Formato, Direção e Copywriting.
    Extrai blocos narrativos dinâmicos, pistas de áudio absolutas, score de clickbait,
    emojis e a nova Super Feature Semântica (palavras_chave).
    """
    print("Iniciando Etapa 5: Processando Engenharia de Formato e Copywriting...")

    try:
        # 1. Timeline do Roteiro (estrutura_blocos) -> Ajuste estático para pipeline linha por linha
        if 'duracao_segundos' in df.columns:
            condicoes = [
                df['duracao_segundos'] <= 20,  # Ultra curtos: Impacto e Loop
                (df['duracao_segundos'] > 20) & (df['duracao_segundos'] <= 45), # Padrão: Historinha rápida
                df['duracao_segundos'] > 45 # Shorts Longos: Narrativa densa (inclui os novos de até 3 min)
            ]
            
            escolhas = ['bloco_unico_impacto', 'hook_desenvolvimento_punchline', 'hook_narrativa_densa_cta']
            df['estrutura_blocos'] = np.select(condicoes, escolhas, default='desconhecido')

        # 2. Sonoplastia (pistas_audio) -> APENAS no texto_falado com tag musical ♪
        def extrair_audio_limpo(texto):
            if pd.isna(texto):
                return ""
            matches = re.findall(r"\[.*?\]|\(.*?\)|♪", str(texto))
            return " ".join(list(dict.fromkeys(matches))) if matches else ""

        if 'texto_falado' in df.columns:
            df['pistas_audio'] = df['texto_falado'].apply(extrair_audio_limpo).str.strip()

        # 3. Copywriting (clickbait_score) -> Captura o Pico de Grito Visual
        def calc_grito_visual(texto):
            if pd.isna(texto) or len(str(texto)) == 0:
                return 0.0
            texto_str = str(texto)
            caps = sum(1 for c in texto_str if c.isupper())
            pontuacao_agressiva = len(re.findall(r"[!?]", texto_str))
            base_valida = sum(1 for c in texto_str if c.isalpha()) + pontuacao_agressiva
            return ((caps + pontuacao_agressiva) / base_valida * 100) if base_valida > 0 else 0.0

        if 'titulo' in df.columns and 'descricao' in df.columns:
            score_titulo = df['titulo'].apply(calc_grito_visual)
            score_desc = df['descricao'].apply(calc_grito_visual)
            df['clickbait_score'] = np.maximum(score_titulo, score_desc)
        elif 'titulo' in df.columns:
            df['clickbait_score'] = df['titulo'].apply(calc_grito_visual)

        # 4. Emoção (vibe_emojis)
        def extrair_emojis(texto):
            texto_str = str(texto)
            if texto_str == 'nan' or not texto_str.strip():
                return ""
            emojis_encontrados = [c['emoji'] for c in emoji.emoji_list(texto_str)]
            return "".join(emojis_encontrados)

        if 'titulo' in df.columns and 'descricao' in df.columns:
            df['vibe_emojis'] = (df['titulo'].apply(extrair_emojis) + df['descricao'].apply(extrair_emojis))

        # 5. SEO & Semântica (palavras_chave) -> Super Feature
        YOUTUBE_CATEGORY_IDS = {
            "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
            "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
            "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
            "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
            "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
            "18": "Short Movies", "21": "Videoblogging", "30": "Movies",
            "31": "Anime/Animation", "32": "Action/Adventure", "33": "Classics",
            "34": "Comedy (Backend/Movies)", "35": "Documentary", "36": "Drama",
            "37": "Family", "38": "Foreign", "39": "Horror", "40": "Sci-Fi/Fantasy",
            "41": "Thriller", "42": "Shorts", "43": "Shows", "44": "Trailers"
        }

        def construir_core_semantico(row):
            from urllib.parse import unquote
            palavras = []

            topicos = row.get("topicos_wikipedia", "[]")
            if isinstance(topicos, str):
                try: 
                    topicos = ast.literal_eval(topicos)
                except (ValueError, SyntaxError): 
                    topicos = []
            if isinstance(topicos, list):
                for url in topicos:
                    termo = unquote(url.split("/")[-1]).replace("_", " ")
                    palavras.append(termo)

            cat_id = str(row.get("categoria_id", ""))
            if cat_id in YOUTUBE_CATEGORY_IDS:
                palavras.append(YOUTUBE_CATEGORY_IDS[cat_id])

            titulo = str(row.get("titulo", ""))
            descricao = str(row.get("descricao", ""))
            palavras.extend(re.findall(r"#(\w+)", titulo))
            palavras.extend(re.findall(r"#(\w+)", descricao))

            tags = row.get("tags", "[]")
            if isinstance(tags, str):
                try: 
                    tags = ast.literal_eval(tags)
                except (ValueError, SyntaxError): 
                    tags = []
            if isinstance(tags, list):
                palavras.extend(tags)

            palavras_limpas = set()
            for p in palavras:
                if p and isinstance(p, str):
                    palavras_limpas.add(p.strip().lower())

            return ", ".join(sorted(list(palavras_limpas)))

        df['palavras_chave'] = df.apply(construir_core_semantico, axis=1)

        print(" SUCESSO ABSOLUTO!")
        print(" -> Novas colunas adicionadas:")
        print("    [Estrutura]: estrutura_blocos")
        print("    [Direção]: pistas_audio, clickbait_score, vibe_emojis")
        print("    [Semântica]: palavras_chave")
        print("\nAs instruções de Direção, Roteiro e SEO Semântico estão prontas para o LLM!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 5: {e}")

    return df

def purificar_texto_e_calcular_ritmo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purifica a transcrição removendo marcadores nativos do YouTube e prepara para NLP.
    Em seguida, calcula o ritmo de fala real (pacing) com base no texto purificado.
    """
    print("Iniciando Etapa 7: Purificação Semântica e Cálculo de Ritmo...")

    try:
        # 1. Purificação Semântica (texto_falado_limpo)
        def purificar_texto(texto):
            if pd.isna(texto):
                return ""
            texto_str = str(texto)

            # Remove marcadores de acessibilidade [], legendas manuais () e músicas ♪
            texto_sem_tags = re.sub(r'\[.*?\]|\(.*?\)|♪', ' ', texto_str)

            # Padroniza para NLP: remove espaços duplos, limpa pontas e joga para minúsculas
            texto_limpo = re.sub(r'\s+', ' ', texto_sem_tags).strip().lower()

            return texto_limpo

        if 'texto_falado' in df.columns:
            df['texto_falado_limpo'] = df['texto_falado'].apply(purificar_texto)

        # 2. Pacing Real (ritmo_palavras_seg)
        # Agora o split() roda em cima do texto limpo. Um vídeo puramente musical,
        # cujo texto_falado era só "[music]", agora terá 0 palavras, refletindo a realidade.
        if 'texto_falado_limpo' in df.columns and 'duracao_segundos' in df.columns:
            contagem_palavras = df['texto_falado_limpo'].fillna("").apply(lambda x: len(str(x).split()))

            df['ritmo_palavras_seg'] = np.where(
                df['duracao_segundos'] > 0, 
                contagem_palavras / df['duracao_segundos'], 
                0.0
            )

        print(" SUCESSO ABSOLUTO!")
        print(" -> Nova coluna criada: texto_falado_limpo")
        print(" -> Nova coluna criada: ritmo_palavras_seg")
        print(" -> Todas as tags de áudio varridas e texto convertido para minúsculas.")
        print("\nO texto e o ritmo estão perfeitamente ajustados para o Embedding!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 7: {e}")

    return df

def unificar_idioma_ingles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unificação de Idioma para NLP (Arquitetura Híbrida sem perda de dados).
    Traduz as variáveis principais de texto para o inglês usando chunking,
    e garante a normalização (minúsculas) nas colunas estruturais.
    """
    # Inicializa o tqdm para o pandas
    tqdm.pandas()

    print("Iniciando Etapa 8: Unificação de Idioma e Normalização para NLP...")

    try:
        tradutor = GoogleTranslator(source='auto', target='en')

        def traduzir_texto_sem_perda(texto):
            if pd.isna(texto) or str(texto).strip() == "":
                return ""

            texto_str = str(texto)

            # Se for pequeno o suficiente, traduz de uma vez
            if len(texto_str) < 4900:
                try:
                    return tradutor.translate(texto_str)
                except Exception:
                    return texto_str

            # CHUNKING (Zero Perda de Dados)
            # Quebra o texto em pedaços de até 4500 caracteres
            pedacos = textwrap.wrap(texto_str, width=4500, break_long_words=False)
            texto_traduzido_final = ""

            for pedaco in pedacos:
                try:
                    pedaco_traduzido = tradutor.translate(pedaco)
                    texto_traduzido_final += pedaco_traduzido + " "
                except Exception:
                    # Se a API falhar no pedaço, cola o original para não perder a informação
                    texto_traduzido_final += pedaco + " "

            return texto_traduzido_final.strip()

        # 1. Título
        print("1/4: Traduzindo os Títulos...")
        if 'titulo' in df.columns:
            df['titulo_en'] = df['titulo'].progress_apply(traduzir_texto_sem_perda)

        # 2. Descrição
        print("\n2/4: Traduzindo as Descrições...")
        if 'descricao' in df.columns:
            df['descricao_en'] = df['descricao'].progress_apply(traduzir_texto_sem_perda)

        # 3. Texto Falado Limpo (Forçando minúsculas para NLP)
        print("\n3/4: Traduzindo o Texto Falado Limpo (Forçando lowercase)...")
        if 'texto_falado_limpo' in df.columns:
            df['texto_falado_limpo_en'] = df['texto_falado_limpo'].progress_apply(traduzir_texto_sem_perda).str.lower()

        # 4. Palavras-Chave (A nova Super Feature, forçando minúsculas para NLP)
        print("\n4/4: Traduzindo as Palavras-Chave (Forçando lowercase)...")
        if 'palavras_chave' in df.columns:
            df['palavras_chave_en'] = df['palavras_chave'].progress_apply(traduzir_texto_sem_perda).str.lower()

        print("\n SUCESSO ABSOLUTO! Idiomas unificados e matriz pronta para Machine Learning.")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 8: {e}")

    return df

def extrair_features_thumbnail_gemini(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extração via API Gemini (Modo Trator + Log de Erro Real + Log Sucesso)
    Usa o Gemini Vision para extrair texto e contexto visual das thumbnails.
    """
    print("🚀 Iniciando Etapa 9: Extração via Gemini 2.5 Flash...")

    # 1. CONFIGURAÇÃO DA API (Adaptado do Colab para o .env local)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(" ⚠️ [Aviso] GEMINI_API_KEY não encontrada no .env. Pulando análise visual.")
        return df
        
    client = genai.Client(api_key=api_key)

    def extrair_dados_gemini(url_imagem):
        """Extração persistente: agora mostrando o erro real no console"""

        # Download da imagem
        try:
            response = requests.get(url_imagem, timeout=10)
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            return {"error": f"Erro download: {str(e)[:30]}"}

        prompt = (
            "Analyze this YouTube thumbnail and return all outputs strictly in English. "
            "1. Identify if there is any text written on the image. "
            "2. Extract all visible text and TRANSLATE it completely to English (do not keep the original language). "
            "3. Summarize the visual scene in a short sentence. "
            "Return a JSON format using exactly these keys: 'has_text' (boolean), 'text_content' (string), and 'visual_summary' (string)."
        )

        # Loop de insistência (Modo Trator)
        while True:
            try:
                res = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                return json.loads(res.text)

            except Exception as e:
                erro_real = str(e)
                erro_clean = erro_real.lower()

                # Mostra o erro exato para o usuário
                print(f"\n ⚠️ Erro na API: {erro_real[:150]}...")

                if any(cod in erro_clean for cod in ["429", "503", "500", "unavailable", "exhausted", "quota"]):
                    print(" ⏳ Cota atingida ou instabilidade. Aguardando 15s para tentar de novo...")
                    time.sleep(15)
                else:
                    print(" 🔄 Tentando novamente em 15s...")
                    time.sleep(15)

    # 2. EXECUÇÃO NO DATAFRAME
    if 'descricao_visual_thumb' not in df.columns:
        df['descricao_visual_thumb'] = ""
        df['texto_thumbnail'] = ""

    indices_validos = df[df['thumb_maxres'].str.startswith("http", na=False)].index
    print(f"📦 Processando {len(indices_validos)} imagens...\n")

    for idx in tqdm(indices_validos, desc="Progresso"):
        
        # Trava de segurança corrigida (Trata None, NaN e "")
        valor_atual = df.at[idx, 'descricao_visual_thumb']
        if pd.notna(valor_atual) and str(valor_atual).strip() != "":
            continue

        url = df.at[idx, 'thumb_maxres']
        resultado = extrair_dados_gemini(url)

        if "error" not in resultado:
            resumo = resultado.get('visual_summary', "")
            texto = resultado.get('text_content', "")

            df.at[idx, 'descricao_visual_thumb'] = resumo
            df.at[idx, 'texto_thumbnail'] = texto

            # LOG DETALHADO DE SUCESSO
            print(f"\n✅ Index {idx} finalizado:")
            print(f"   🖼️ Resumo: {resumo}")
            print(f"   📝 Texto:  {texto}")
        else:
            print(f"\n 🚫 [{idx}] Ignorado (Falha Técnica): {resultado['error']}")

        # Pausa estratégica para manter ~15 requisições por minuto
        time.sleep(10)

    print("\n✅ Etapa 9 Finalizada!")
    return df

def maestro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestra a execução sequencial de todas as 9 etapas da Usina de Features.
    Recebe o DataFrame (a linha da Salinha VIP) bruto e devolve processado
    com a ordem exata das 41 colunas oficiais, printando o antes e depois.
    """
    print("\n" + "="*55)
    print("🎬 INICIANDO USINA DE FEATURES (MAESTRO)")
    print("="*55)

    # Etapa 1: Limpeza Estrutural
    df = remover_colunas_inuteis(df)
    
    # Etapa 2: Tratamento de Nulos
    df = tratar_valores_nulos(df)
    
    # Etapa 3: Limpeza de Erros de Raspagem
    df = limpar_erros_transcricao(df)
    
    # Etapa 4: Tração e Metadados Temporais
    df = criar_features_tracao_metadados(df)
    
    # Etapa 5: Formato, Direção e Copywriting
    df = criar_features_formato_copywriting(df)
    
    # Etapa 7: Purificação do Texto Falado
    df = purificar_texto_e_calcular_ritmo(df)
    
    # Etapa 8: Unificação de Idioma (Tradução segura)
    df = unificar_idioma_ingles(df)
    
    # Etapa 9: Visão Computacional (Gemini 2.5 Flash)
    df = extrair_features_thumbnail_gemini(df)

    # ==============================================================
    # ETAPA 10: REORDENAÇÃO FINAL (ENFORCING DO SCHEMA OFICIAL)
    # ==============================================================
    print("\n" + "-"*55)
    print("ETAPA 10: ALINHAMENTO DE COLUNAS")
    print("-"*55)
    
    # PRINT ANTES: Mostra a bagunça que o Pandas fez
    print(f"ORDEM ANTES:\n{df.columns.tolist()}\n")

    ordem_oficial = [
        'video_id', 'titulo', 'descricao', 'tags', 'texto_falado',
        'canal_id', 'canal_nome', 'visualizacoes', 'curtidas', 'comentarios',
        'data_publicacao', 'duracao_segundos', 'tem_legenda_nativa',
        'conteudo_licenciado', 'feito_para_criancas', 'categoria_id',
        'idioma_audio_default', 'idioma_texto_default', 'topicos_wikipedia',
        'thumb_maxres', 'nicho', 'dia_postagem', 'hora_postagem', 'idade_dias',
        'velocidade_views', 'taxa_conversao', 'taxa_discussao', 'score_viral',
        'estrutura_blocos', 'ritmo_palavras_seg', 'pistas_audio', 'clickbait_score',
        'vibe_emojis', 'palavras_chave', 'texto_falado_limpo', 'texto_falado_limpo_en',
        'titulo_en', 'descricao_en', 'descricao_visual_thumb', 'texto_thumbnail',
        'palavras_chave_en'
    ]
    
    # Filtra as colunas para evitar o erro de 'KeyError' caso alguma falhe
    colunas_presentes = [col for col in ordem_oficial if col in df.columns]
    
    # Salva qualquer coluna extra que tenha sido gerada
    colunas_extras = [col for col in df.columns if col not in ordem_oficial]
    
    # Aplica a máscara de ordenação (oficiais primeiro, sobras depois)
    df = df[colunas_presentes + colunas_extras]

    # PRINT DEPOIS: Mostra a lista perfeitamente alinhada
    print(f" ORDEM DEPOIS:\n{df.columns.tolist()}")

    print("\n" + "="*55)
    print(" USINA DE FEATURES FINALIZADA E COLUNAS ORDENADAS!")
    print("="*55 + "\n")

    return df