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
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from calibrar_limiares import carregar_limiares
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
    Calcula idade dinâmica, velocidade de views e taxas de engajamento.
    O score_viral é calculado em etapa posterior, após estrutura_blocos e ritmo_palavras_seg existirem.
    """
    print("Iniciando Etapa 4: Processando Engenharia de Tração e Metadados...")

    try:
        # 1. Metadados Temporais (Dia e Hora)
        df['data_publicacao_dt'] = pd.to_datetime(df['data_publicacao'], utc=True)

        df['dia_postagem']  = df['data_publicacao_dt'].dt.day_name().str.lower()
        df['hora_postagem'] = df['data_publicacao_dt'].dt.hour

        # 2. Segurança de Tipagem
        for col in ['visualizacoes', 'curtidas', 'comentarios']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 3. Idade Dinâmica
        # Piso mínimo de 1 evita divisão por zero na velocidade_views.
        # Teto fixo de 365 dias evita que vídeos muito antigos distorçam a métrica
        # independente do nicho ou volume do dataset.
        data_atual  = datetime.now(timezone.utc)
        idade_bruta = (data_atual - df['data_publicacao_dt']).dt.days
        df['idade_dias'] = np.clip(idade_bruta, 1, 365)

        # 4. Velocidade de Views
        df['velocidade_views'] = df['visualizacoes'] / df['idade_dias']

        # 5. Taxas de Engajamento
        df['taxa_conversao'] = np.where(
            df['visualizacoes'] > 0,
            (df['curtidas']    / df['visualizacoes']) * 100,
            0
        )
        df['taxa_discussao'] = np.where(
            df['visualizacoes'] > 0,
            (df['comentarios'] / df['visualizacoes']) * 100,
            0
        )

        # 6. Janela de Postagem
        # Substitui hora_postagem crua como feature preditiva isolada.
        # Faixas baseadas no comportamento real do nicho:
        # .log: 80% das postagens entre 3h e 14h — madrugada e manhã dominam.
        hora = df['hora_postagem']
        condicoes = [hora <= 5, hora <= 11, hora <= 17]
        escolhas  = ['madrugada', 'manha', 'tarde']
        df['janela_postagem'] = np.select(condicoes, escolhas, default='noite')

        # Limpeza: coluna datetime temporária não vai para o CSV final
        df = df.drop(columns=['data_publicacao_dt'], errors='ignore')

        print(" SUCESSO!")
        print("    [Tempo]  : dia_postagem, hora_postagem, janela_postagem, idade_dias")
        print("    [Tração] : velocidade_views, taxa_conversao, taxa_discussao")
        print("    NOTA     : score_viral calculado em etapa posterior.")

    except Exception as e:
        print(f" Erro na Etapa 4: {e}")
        print("Dica: Verifique se a coluna 'data_publicacao' e as colunas numéricas estão corretas.")

    return df

def criar_features_formato_copywriting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa Engenharia de Formato, Direção e Copywriting.
    Extrai pistas de áudio, emojis e a Super Feature Semântica (palavras_chave).
    NOTA: estrutura_blocos calculada em etapa posterior, após sinais de roteiro e áudio existirem.
    NOTA: clickbait_score calculado em etapa posterior, após tradução e Gemini.
    """
    print("Iniciando Etapa 5: Processando Engenharia de Formato e Copywriting...")

    try:
        # 1. Pistas de Áudio (pistas_audio)
        # Extrai marcadores nativos do YouTube: [música], [risadas], ♪ etc.
        # Preservado como derivada do texto_falado para evitar recomputação.
        def extrair_audio_limpo(texto):
            if pd.isna(texto):
                return ""
            matches = re.findall(r"\[.*?\]|\(.*?\)|♪", str(texto))
            return " ".join(list(dict.fromkeys(matches))) if matches else ""

        if 'texto_falado' in df.columns:
            df['pistas_audio'] = df['texto_falado'].apply(extrair_audio_limpo).str.strip()

        # 2. Emoção Visual (vibe_emojis)
        # Deduplicação com set() para evitar inflação de emojis que aparecem
        # tanto no título quanto na descrição.
        def extrair_emojis(texto):
            texto_str = str(texto)
            if texto_str == 'nan' or not texto_str.strip():
                return set()
            return set(c['emoji'] for c in emoji.emoji_list(texto_str))

        if 'titulo' in df.columns and 'descricao' in df.columns:
            df['vibe_emojis'] = df.apply(
                lambda row: "".join(sorted(
                    extrair_emojis(row['titulo']) | extrair_emojis(row['descricao'])
                )),
                axis=1
            )

        # 3. SEO & Semântica (palavras_chave) — Super Feature
        # Consolida: tópicos Wikipedia + categoria YouTube + hashtags + tags
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

            titulo   = str(row.get("titulo", ""))
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

        print(" SUCESSO!")
        print("    [Direção]   : pistas_audio, vibe_emojis")
        print("    [Semântica] : palavras_chave")
        print("    NOTA        : estrutura_blocos calculada em etapa posterior.")
        print("    NOTA        : clickbait_score calculado em etapa posterior.")

    except Exception as e:
        print(f" Erro na Etapa 5: {e}")

    return df

def purificar_texto_e_calcular_ritmo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purifica a transcrição removendo marcadores nativos do YouTube e prepara para NLP.
    Gera apenas texto_falado_limpo nesta etapa.
    NOTA: ritmo_palavras_seg calculado em etapa posterior, após tradução,
    pois usa texto_falado_limpo_en para padronizar o cálculo em um único idioma.
    """
    print("Iniciando Etapa 7: Purificação Semântica do Texto Falado...")

    try:
        # Purificação Semântica (texto_falado_limpo)
        # Remove marcadores de acessibilidade [], legendas manuais () e músicas ♪
        # que o YouTube insere automaticamente nas transcrições.
        # Um vídeo puramente musical cujo texto_falado era só "[music]"
        # ficará com string vazia, refletindo a realidade do conteúdo falado.
        def purificar_texto(texto):
            if pd.isna(texto):
                return ""
            texto_str = str(texto)
            texto_sem_tags = re.sub(r'\[.*?\]|\(.*?\)|♪', ' ', texto_str)
            texto_limpo    = re.sub(r'\s+', ' ', texto_sem_tags).strip().lower()
            return texto_limpo

        if 'texto_falado' in df.columns:
            df['texto_falado_limpo'] = df['texto_falado'].apply(purificar_texto)

        print(" SUCESSO!")
        print("    [Texto] : texto_falado_limpo")
        print("    NOTA    : ritmo_palavras_seg calculado em etapa posterior (requer texto_falado_limpo_en).")

    except Exception as e:
        print(f" Erro na Etapa 7: {e}")

    return df

def unificar_idioma_ingles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unificação de Idioma para NLP (Arquitetura Híbrida sem perda de dados).
    Traduz as variáveis principais de texto para o inglês usando chunking,
    garantindo normalização (minúsculas) nas colunas estruturais.
    Inclui tradução de pistas_audio para padronizar marcadores sonoros
    entre nichos de idiomas diferentes.
    """
    tqdm.pandas()
    print("Iniciando Etapa 8: Unificação de Idioma e Normalização para NLP...")

    try:
        tradutor = GoogleTranslator(source='auto', target='en')

        def traduzir_texto_sem_perda(texto):
            if pd.isna(texto) or str(texto).strip() == "":
                return ""

            texto_str = str(texto)

            # Textos pequenos: traduz de uma vez
            if len(texto_str) < 4900:
                try:
                    return tradutor.translate(texto_str)
                except Exception:
                    return texto_str

            # CHUNKING: quebra em pedaços de até 4500 chars para não perder dados
            pedacos = textwrap.wrap(texto_str, width=4500, break_long_words=False)
            texto_traduzido_final = ""
            for pedaco in pedacos:
                try:
                    texto_traduzido_final += tradutor.translate(pedaco) + " "
                except Exception:
                    # Se a API falhar no pedaço, preserva o original
                    texto_traduzido_final += pedaco + " "

            return texto_traduzido_final.strip()

        # 1. Título
        print("1/5: Traduzindo os Títulos...")
        if 'titulo' in df.columns:
            df['titulo_en'] = df['titulo'].progress_apply(traduzir_texto_sem_perda)

        # 2. Descrição
        print("\n2/5: Traduzindo as Descrições...")
        if 'descricao' in df.columns:
            df['descricao_en'] = df['descricao'].progress_apply(traduzir_texto_sem_perda)

        # 3. Texto Falado Limpo
        print("\n3/5: Traduzindo o Texto Falado Limpo (lowercase)...")
        if 'texto_falado_limpo' in df.columns:
            df['texto_falado_limpo_en'] = df['texto_falado_limpo'].progress_apply(traduzir_texto_sem_perda).str.lower()

        # 4. Palavras-Chave
        print("\n4/5: Traduzindo as Palavras-Chave (lowercase)...")
        if 'palavras_chave' in df.columns:
            df['palavras_chave_en'] = df['palavras_chave'].progress_apply(traduzir_texto_sem_perda).str.lower()

        # 5. Pistas de Áudio
        # Traduz marcadores como [हंसी] → [laughter], [संगीत] → [music]
        # para padronizar entre nichos de idiomas diferentes.
        print("\n5/5: Traduzindo as Pistas de Áudio (lowercase)...")
        if 'pistas_audio' in df.columns:
            df['pistas_audio_en'] = df['pistas_audio'].progress_apply(traduzir_texto_sem_perda).str.lower()

        print("\n SUCESSO! Idiomas unificados e matriz pronta para as próximas etapas.")
        print("    titulo_en, descricao_en, texto_falado_limpo_en, palavras_chave_en, pistas_audio_en")

    except Exception as e:
        print(f" Erro na Etapa 8: {e}")

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

def criar_features_faixa_duracao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria label categórica de faixa de duração para Shorts.
    Usa os limites técnicos reais do YouTube Shorts como referência:
    - ultra_curto  : até 30s  — impacto imediato, loop natural
    - short_padrao : 31–60s   — formato mais comum, hook + punchline
    - short_longo  : 61–180s  — narrativa densa, limite máximo do Shorts
    Complementa estrutura_blocos com uma âncora de tempo limpa e legível.
    """
    print("Iniciando Etapa 5b: Criando Faixa de Duração...")

    try:
        if 'duracao_segundos' not in df.columns:
            print(" Coluna 'duracao_segundos' não encontrada. Pulando etapa.")
            return df

        condicoes = [
            df['duracao_segundos'] <= 30,
            (df['duracao_segundos'] > 30) & (df['duracao_segundos'] <= 60),
            df['duracao_segundos'] > 60
        ]
        escolhas = ['ultra_curto', 'short_padrao', 'short_longo']

        df['faixa_duracao'] = np.select(condicoes, escolhas, default='desconhecido')

        # Distribuição para conferência rápida
        distribuicao = df['faixa_duracao'].value_counts()
        print(" SUCESSO!")
        print("    [Formato] : faixa_duracao")
        print("    Distribuição:")
        for faixa, count in distribuicao.items():
            print(f"      {faixa:<20}: {count} vídeos ({count/len(df)*100:.1f}%)")

    except Exception as e:
        print(f" Erro na Etapa 5b: {e}")

    return df

def calcular_densidade_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    densidade_roteiro — total de palavras no texto_falado_limpo_en.
    Usa _en para contagem comparável entre vídeos de idiomas diferentes.
    .md: min 2 | max 647 | mediana 97 palavras.
    Consumido por: calcular_estrutura_blocos.
    Depende de: texto_falado_limpo_en (unificar_idioma_ingles).
    """
    print("Iniciando densidade_roteiro...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        df['densidade_roteiro'] = df['texto_falado_limpo_en'].fillna("").apply(
            lambda x: len(str(x).split())
        )

        sem_roteiro = (df['densidade_roteiro'] == 0).sum()
        mediana     = df['densidade_roteiro'][df['densidade_roteiro'] > 0].median()

        print(" SUCESSO!")
        print("    [Roteiro] : densidade_roteiro")
        print(f"    Vídeos sem roteiro (0 palavras) : {sem_roteiro} ({sem_roteiro/len(df)*100:.1f}%)")
        print(f"    Mediana entre vídeos com roteiro: {mediana:.0f} palavras")

    except Exception as e:
        print(f" Erro em densidade_roteiro: {e}")

    return df

def calcular_tem_repeticao_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    tem_repeticao_roteiro — True se alguma palavra de conteúdo aparece 5x+
    no texto_falado_limpo_en (após remoção de stopwords em inglês).
    Threshold 5x captura hook intencional sem confundir com recorrência
    natural de vocabulário num roteiro de ~97 palavras (mediana do nicho).
    .log: 31.9% dos textos têm repetição — técnica de hook em Shorts.
    Consumido por: calcular_estrutura_blocos.
    Depende de: texto_falado_limpo_en (unificar_idioma_ingles).
    """
    print("Iniciando tem_repeticao_roteiro...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))

        def _tem_repeticao(t):
            if pd.isna(t) or not str(t).strip():
                return False
            palavras = re.findall(r'\b[a-z]{3,}\b', str(t).lower())
            relevantes = [p for p in palavras if p not in stop_words]
            if not relevantes:
                return False
            return bool(pd.Series(relevantes).value_counts().iloc[0] >= 5)

        df['tem_repeticao_roteiro'] = df['texto_falado_limpo_en'].apply(_tem_repeticao)

        pct = df['tem_repeticao_roteiro'].mean() * 100
        print(" SUCESSO!")
        print("    [Roteiro] : tem_repeticao_roteiro")
        print(f"    Vídeos com repetição (≥5x) : {df['tem_repeticao_roteiro'].sum()} ({pct:.1f}%)")

    except Exception as e:
        print(f" Erro em tem_repeticao_roteiro: {e}")

    return df

def calcular_gancho_primeira_frase(df: pd.DataFrame) -> pd.DataFrame:
    """
    gancho_primeira_frase — primeiras 15 palavras de texto_falado_limpo_en.
    Isola o hook de abertura — em Shorts os primeiros ~3s determinam retenção.
    Uso direto: comparar abertura dos virais vs medianos e gerar templates replicáveis.
    Depende de: texto_falado_limpo_en (unificar_idioma_ingles).
    """
    print("Iniciando gancho_primeira_frase...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        def _gancho(t):
            if pd.isna(t) or not str(t).strip():
                return ""
            return " ".join(str(t).split()[:15])

        df['gancho_primeira_frase'] = df['texto_falado_limpo_en'].apply(_gancho)

        sem_gancho = (df['gancho_primeira_frase'] == "").sum()
        print(" SUCESSO!")
        print("    [Roteiro] : gancho_primeira_frase (primeiras 15 palavras)")
        print(f"    Vídeos sem gancho (mudos/musicais): {sem_gancho} ({sem_gancho/len(df)*100:.1f}%)")

    except Exception as e:
        print(f" Erro em gancho_primeira_frase: {e}")

    return df

def calcular_sentimento_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    sentimento_roteiro — tom emocional do roteiro: positivo | negativo | neutro.
    Usa nltk VADER (léxico otimizado para texto informal/social media em inglês).
    Limiares padrão VADER: compound >= 0.05 → positivo | <= -0.05 → negativo | entre → neutro.
    Analisa texto_falado_limpo_en — comparável entre vídeos de idiomas diferentes.
    Depende de: texto_falado_limpo_en (unificar_idioma_ingles).
    """
    print("Iniciando sentimento_roteiro...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            print("    Baixando léxico VADER (primeira execução)...")
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()

        def _sentimento(t):
            if pd.isna(t) or not str(t).strip():
                return "neutro"
            compound = sia.polarity_scores(str(t))['compound']
            if compound >= 0.05:
                return "positivo"
            if compound <= -0.05:
                return "negativo"
            return "neutro"

        df['sentimento_roteiro'] = df['texto_falado_limpo_en'].apply(_sentimento)

        dist = df['sentimento_roteiro'].value_counts().to_dict()
        print(" SUCESSO!")
        print("    [Roteiro] : sentimento_roteiro")
        print(f"    Distribuição: {dist}")

    except Exception as e:
        print(f" Erro em sentimento_roteiro: {e}")

    return df

def calcular_tipo_audio_dominante(df: pd.DataFrame) -> pd.DataFrame:
    """
    tipo_audio_dominante — tag de áudio mais frequente em pistas_audio_en.
    As pistas_audio_en são os marcadores automáticos do YouTube ([music],
    [laughter] etc.) extraídos do texto_falado e traduzidos para inglês.
    Retorna a tag mais frequente como string limpa, ou "" quando ausente.
    Consumido por: calcular_estrutura_blocos.
    Depende de: pistas_audio_en (unificar_idioma_ingles).
    """
    print("Iniciando tipo_audio_dominante...")

    try:
        if 'pistas_audio_en' not in df.columns:
            print(" 'pistas_audio_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        def _dominante(t):
            if pd.isna(t) or not str(t).strip():
                return ""
            tags = re.findall(r'\[.*?\]|\(.*?\)|♪', str(t))
            if not tags:
                return ""
            return pd.Series(tags).value_counts().index[0]

        df['tipo_audio_dominante'] = df['pistas_audio_en'].apply(_dominante)

        dist = df['tipo_audio_dominante'].value_counts().to_dict()
        sem_audio = (df['tipo_audio_dominante'] == "").sum()
        print(" SUCESSO!")
        print("    [Áudio] : tipo_audio_dominante")
        print(f"    Vídeos sem pistas : {sem_audio} ({sem_audio/len(df)*100:.1f}%)")
        print(f"    Distribuição      : {dist}")

    except Exception as e:
        print(f" Erro em tipo_audio_dominante: {e}")

    return df

def calcular_estrutura_blocos(df: pd.DataFrame) -> pd.DataFrame:
    """
    estrutura_blocos — formato narrativo real do vídeo.
    Cada label recebe um score de compatibilidade calculado a partir dos
    5 sinais em conjunto. O label com maior score vence.
    Limiares calculados por quartis do dataset — universal.
    Labels: impacto_rapido | esquete | esquete_com_hook | narrativa
    Depende de: faixa_duracao, ritmo_palavras_seg, densidade_roteiro,
                tem_repeticao_roteiro, tipo_audio_dominante.
    """
    print("Iniciando estrutura_blocos...")

    try:
        colunas = ['faixa_duracao', 'ritmo_palavras_seg', 'densidade_roteiro',
                   'tem_repeticao_roteiro', 'tipo_audio_dominante']
        faltando = [c for c in colunas if c not in df.columns]
        if faltando:
            print(f" Colunas ausentes: {faltando}. Execute as etapas anteriores primeiro.")
            return df

        nicho_atual = df['nicho'].iloc[0]
        limiares = carregar_limiares()[nicho_atual]['estrutura_blocos']
        q1_den = limiares['densidade_roteiro']['q1']
        q3_den = limiares['densidade_roteiro']['q3']
        q1_rit = limiares['ritmo_palavras_seg']['q1']
        q3_rit = limiares['ritmo_palavras_seg']['q3']

        def _score(row):
            faixa     = row['faixa_duracao']
            ritmo     = row['ritmo_palavras_seg']
            density   = row['densidade_roteiro']
            repeticao = row['tem_repeticao_roteiro']
            audio     = str(row['tipo_audio_dominante']).strip()

            scores = {
                'impacto_rapido'  : 0.0,
                'esquete'         : 0.0,
                'esquete_com_hook': 0.0,
                'narrativa'       : 0.0,
            }

            # faixa_duracao
            if faixa == 'ultra_curto':
                scores['impacto_rapido']   += 0.25
            elif faixa == 'short_padrao':
                scores['esquete']          += 0.15
                scores['esquete_com_hook'] += 0.15
            else:
                scores['narrativa']        += 0.25

            # ritmo_palavras_seg
            if ritmo >= q3_rit:
                scores['impacto_rapido']   += 0.20
                scores['esquete_com_hook'] += 0.10
            elif ritmo <= q1_rit:
                scores['narrativa']        += 0.20
                scores['esquete']          += 0.10
            else:
                scores['esquete']          += 0.15
                scores['esquete_com_hook'] += 0.15

            # densidade_roteiro
            if density <= q1_den:
                scores['impacto_rapido']   += 0.20
            elif density >= q3_den:
                scores['narrativa']        += 0.20
            else:
                scores['esquete']          += 0.15
                scores['esquete_com_hook'] += 0.15

            # tem_repeticao_roteiro
            if repeticao:
                scores['esquete_com_hook'] += 0.20
                scores['impacto_rapido']   += 0.05
            else:
                scores['esquete']          += 0.10
                scores['narrativa']        += 0.10

            # tipo_audio_dominante
            if audio:
                scores['impacto_rapido']   += 0.15
                scores['esquete_com_hook'] += 0.05
            else:
                scores['narrativa']        += 0.10
                scores['esquete']          += 0.10

            return max(scores, key=scores.get)

        df['estrutura_blocos'] = df.apply(_score, axis=1)

        dist = df['estrutura_blocos'].value_counts().to_dict()
        print(" SUCESSO!")
        print("    [Formato] : estrutura_blocos")
        print(f"    Limiares  : densidade Q1={q1_den:.0f} Q3={q3_den:.0f} | "
              f"ritmo Q1={q1_rit:.2f} Q3={q3_rit:.2f}")
        print(f"    Distribuição: {dist}")

    except Exception as e:
        print(f" Erro em estrutura_blocos: {e}")

    return df

def calcular_ritmo_palavras_seg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o ritmo de fala real (pacing) em palavras por segundo.
    Usa texto_falado_limpo_en como base para padronizar o cálculo em inglês,
    tornando o ritmo comparável entre vídeos de qualquer idioma.
    Roda após unificar_idioma_ingles (etapa 8), pois depende de texto_falado_limpo_en.
    
    Vídeos puramente musicais ou sem fala terão ritmo 0.0, refletindo a realidade
    do conteúdo — o texto limpo já estará vazio após a purificação.
    """
    print("Iniciando Etapa 10: Calculando Ritmo de Palavras por Segundo...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" Coluna 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        if 'duracao_segundos' not in df.columns:
            print(" Coluna 'duracao_segundos' não encontrada. Pulando etapa.")
            return df

        contagem_palavras = (
            df['texto_falado_limpo_en']
            .fillna("")
            .apply(lambda x: len(str(x).split()))
        )

        df['ritmo_palavras_seg'] = np.where(
            df['duracao_segundos'] > 0,
            contagem_palavras / df['duracao_segundos'],
            0.0
        )

        # Métricas para conferência rápida
        sem_fala = (df['ritmo_palavras_seg'] == 0).sum()
        mediana  = df['ritmo_palavras_seg'][df['ritmo_palavras_seg'] > 0].median()

        print(" SUCESSO!")
        print("    [Ritmo] : ritmo_palavras_seg")
        print(f"    Vídeos sem fala (ritmo = 0) : {sem_fala} ({sem_fala/len(df)*100:.1f}%)")
        print(f"    Mediana entre vídeos com fala: {mediana:.2f} palavras/seg")

    except Exception as e:
        print(f" Erro na Etapa 10: {e}")

    return df

def calcular_score_viral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o score viral normalizado e o label categórico de viralidade.

    Métricas do score (todas normalizadas via Min-Max antes de combinar):
    - velocidade_views : views por dia de vida do vídeo (peso 0.60)
    - taxa_conversao   : % de curtidas sobre views     (peso 0.30)
    - taxa_discussao   : % de comentários sobre views  (peso 0.10)

    Pesos fixos calibrados para o nicho de Shorts:
    - views domina pois é o que o algoritmo usa pra distribuir
    - curtidas são o principal sinal de qualidade em Shorts
    - comentários são raros em Shorts mas ainda informativos

    Roda após: criar_features_tracao_metadados (velocidade_views, taxas)
    """
    print("Iniciando Etapa 11: Calculando Score Viral...")

    try:
        colunas_necessarias = ['velocidade_views', 'taxa_conversao', 'taxa_discussao']
        faltando = [c for c in colunas_necessarias if c not in df.columns]
        if faltando:
            print(f" Colunas ausentes: {faltando}. Execute as etapas anteriores primeiro.")
            return df

        nicho_atual = df['nicho'].iloc[0]
        limiares_score = carregar_limiares()[nicho_atual]['score_viral']

        def minmax(serie, nome_coluna):
            min_val = limiares_score[nome_coluna]['min']
            max_val = limiares_score[nome_coluna]['max']
            if max_val == min_val:
                return pd.Series(0.0, index=serie.index)
            return (serie - min_val) / (max_val - min_val)

        views_norm     = minmax(df['velocidade_views'], 'velocidade_views')
        conversao_norm = minmax(df['taxa_conversao'], 'taxa_conversao')
        discussao_norm = minmax(df['taxa_discussao'], 'taxa_discussao')

        peso_views, peso_conversao, peso_discussao = 0.60, 0.30, 0.10

        print("    Pesos fixos calibrados para Shorts:")
        print(f"      velocidade_views : {peso_views:.0%}")
        print(f"      taxa_conversao   : {peso_conversao:.0%}")
        print(f"      taxa_discussao   : {peso_discussao:.0%}")

        df['score_viral'] = (
            (views_norm     * peso_views)     +
            (conversao_norm * peso_conversao) +
            (discussao_norm * peso_discussao)
        )

        cortes = carregar_limiares()[nicho_atual]['label_viral']['cortes_score_viral']

        def classificar_label(score):
            if score <= cortes['q25']:
                return 'frio'
            elif score <= cortes['q50']:
                return 'aquecido'
            elif score <= cortes['q75']:
                return 'viral'
            else:
                return 'super_viral'

        df['label_viral'] = df['score_viral'].apply(classificar_label)

        distribuicao = df['label_viral'].value_counts().sort_index()
        print(" SUCESSO!")
        print("    [Score] : score_viral (0–1, normalizado)")
        print("    [Label] : label_viral")
        print("    Distribuição:")
        for label, count in distribuicao.items():
            print(f"      {label:<15}: {count} vídeos ({count/len(df)*100:.1f}%)")

    except Exception as e:
        print(f" Erro na Etapa 11: {e}")

    return df

def calcular_clickbait_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o score de atração visual e textual do vídeo.

    Combina três pilares do clique no YouTube:
    1. Título      : titulo_en       — emojis, hashtags, maiúsculas, !?, comprimento
    2. Thumbnail   : texto_thumbnail  — presença e tamanho do hook visual escrito na capa
                     descricao_visual_thumb — riqueza da cena visual descrita
    3. Descrição   : descricao_en    — hashtags, emojis, força da primeira linha

    Todos os sinais são normalizados via Min-Max antes de combinar,
    garantindo que nenhum pilar domine por diferença de escala.
    Funciona em qualquer idioma pois usa as colunas _en como base.

    Roda após: unificar_idioma_ingles (etapa 8)
               extrair_features_thumbnail_gemini (etapa 9)
    """
    print("Iniciando Etapa 12: Calculando Clickbait Score...")

    try:
        colunas_necessarias = [
            'titulo_en', 'descricao_en',
            'texto_thumbnail', 'descricao_visual_thumb'
        ]
        faltando = [c for c in colunas_necessarias if c not in df.columns]
        if faltando:
            print(f" Colunas ausentes: {faltando}. Execute as etapas anteriores primeiro.")
            return df

        def minmax(serie):
            min_val = serie.min()
            max_val = serie.max()
            if max_val == min_val:
                return pd.Series(0.0, index=serie.index)
            return (serie - min_val) / (max_val - min_val)

        # ── PILAR 1: Título (titulo_en) ──────────────────────────────────────

        def score_titulo(texto):
            if pd.isna(texto) or not str(texto).strip():
                return 0.0
            t = str(texto)

            # Emojis: presença de elementos visuais no título
            qtd_emojis = len(emoji.emoji_list(t))

            # Hashtags: intenção de alcance (#shorts, #viral, #trending etc.)
            qtd_hashtags = len(re.findall(r"#\w+", t))

            # Maiúsculas e pontuação agressiva: válido pois o texto está em inglês
            chars_alpha = sum(1 for c in t if c.isalpha())
            caps = sum(1 for c in t if c.isupper())
            pontuacao = len(re.findall(r"[!?]", t))
            ratio_caps = (caps + pontuacao) / (chars_alpha + pontuacao) if (chars_alpha + pontuacao) > 0 else 0.0

            # Comprimento: títulos entre 40–70 chars tendem a performar melhor
            # (visíveis no mobile sem cortar, informativos o suficiente)
            comprimento = len(t)
            score_comprimento = 1.0 if 40 <= comprimento <= 70 else max(0.0, 1.0 - abs(comprimento - 55) / 55)

            return qtd_emojis + qtd_hashtags + ratio_caps + score_comprimento

        score_titulo_raw = df['titulo_en'].apply(score_titulo)

        # ── PILAR 2: Thumbnail ───────────────────────────────────────────────

        def score_thumbnail(texto_thumb, desc_visual):
            pontos = 0.0

            # texto_thumbnail: presença e tamanho do hook escrito na capa
            t = str(texto_thumb) if not pd.isna(texto_thumb) else ""
            if t.strip():
                pontos += 1.0  # tem texto na capa
                # Hook curto e direto é mais impactante (até 40 chars = ideal)
                pontos += max(0.0, 1.0 - max(0, len(t) - 40) / 40)

            # descricao_visual_thumb: riqueza da cena descrita pelo Gemini
            # Descrições mais longas indicam cenas mais ricas e detalhadas
            d = str(desc_visual) if not pd.isna(desc_visual) else ""
            if d.strip():
                pontos += min(1.0, len(d) / 100)

            return pontos

        score_thumb_raw = df.apply(
            lambda row: score_thumbnail(row['texto_thumbnail'], row['descricao_visual_thumb']),
            axis=1
        )

        # ── PILAR 3: Descrição (descricao_en) ────────────────────────────────

        def score_descricao(texto):
            if pd.isna(texto) or not str(texto).strip():
                return 0.0
            t = str(texto)

            # Hashtags na descrição: ampliam alcance e sinalizam contexto
            qtd_hashtags = len(re.findall(r"#\w+", t))

            # Emojis na descrição: reforço emocional e visual
            qtd_emojis = len(emoji.emoji_list(t))

            # Força da primeira linha: o que aparece antes do "ver mais"
            # Primeiras 100 chars — quanto mais densa, mais atrai o clique
            primeira_linha = t[:100].strip()
            score_primeira_linha = min(1.0, len(primeira_linha) / 100)

            return qtd_hashtags + qtd_emojis + score_primeira_linha

        score_desc_raw = df['descricao_en'].apply(score_descricao)

        # ── Score Final: média normalizada dos três pilares ──────────────────
        # Peso igual para os três pilares — cada um representa uma etapa
        # da jornada do clique: thumbnail → título → descrição.
        df['clickbait_score'] = (
            minmax(score_titulo_raw) +
            minmax(score_thumb_raw)  +
            minmax(score_desc_raw)
        ) / 3.0

        mediana = df['clickbait_score'].median()
        print(" SUCESSO!")
        print("    [Atração] : clickbait_score (0–1, normalizado)")
        print(f"    Mediana   : {mediana:.3f}")
        print("    Pilares   : título | thumbnail | descrição")

    except Exception as e:
        print(f" Erro na Etapa 12: {e}")

    return df

def calcular_completude_seo(df: pd.DataFrame) -> pd.DataFrame:
    """
    completude_seo — score contínuo de intensidade de SEO e copywriting.
    Combina presença estrutural e sinais de atração em titulo_en e descricao_en.
    Sinais: descricao preenchida, tags, hashtags, menções @, pontuação !?,
            emojis — todos somados em score único.
    Depende de: titulo_en, descricao_en (unificar_idioma_ingles), tags (original).
    """
    print("Iniciando completude_seo...")

    try:
        def _score(row):
            pontos = 0.0

            titulo = str(row.get('titulo_en', ''))
            desc   = str(row.get('descricao_en', ''))

            # Presença estrutural
            if desc.strip() and desc.strip().lower() != 'nan':
                pontos += 1

            tags = row.get('tags', '[]')
            if isinstance(tags, str):
                try:
                    tags = ast.literal_eval(tags)
                except (ValueError, SyntaxError):
                    tags = []
            if isinstance(tags, list) and len(tags) > 0:
                pontos += 1

            # Sinais de atração — titulo + descricao
            texto_completo = titulo + " " + desc
            pontos += len(re.findall(r'#\w+', texto_completo))
            pontos += len(re.findall(r'@\w+', texto_completo))
            pontos += len(re.findall(r'[!?]', texto_completo))
            pontos += len(emoji.emoji_list(texto_completo))

            return pontos

        df['completude_seo'] = df.apply(_score, axis=1)

        mediana = df['completude_seo'].median()
        maximo  = df['completude_seo'].max()
        print(" SUCESSO!")
        print("    [SEO] : completude_seo (score contínuo)")
        print(f"    Mediana: {mediana:.1f} | Máximo: {maximo:.0f}")

    except Exception as e:
        print(f" Erro em completude_seo: {e}")

    return df

def criar_vocabulario_falado(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Extrai o vocabulário mais relevante do conteúdo falado em cada vídeo.
    Usa texto_falado_limpo_en (já em inglês e normalizado) como base,
    remove stopwords em inglês via nltk e retorna as top_n palavras
    mais frequentes como uma string separada por espaços.

    Complementa palavras_chave_en com o vocabulário real da fala —
    o que o criador disse, não só o que declarou nas tags e hashtags.

    Roda após: unificar_idioma_ingles (etapa 8)
    """

    print("Iniciando Etapa 13: Extraindo Vocabulário Falado...")

    try:
        if 'texto_falado_limpo_en' not in df.columns:
            print(" Coluna 'texto_falado_limpo_en' não encontrada. Execute unificar_idioma_ingles primeiro.")
            return df

        # Garante que as stopwords estão disponíveis localmente.
        # Na primeira execução faz o download automaticamente.
        # Nas seguintes, o nltk detecta que já existe e não baixa de novo.
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            print("    Baixando stopwords do nltk (primeira execução)...")
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))

        def extrair_vocabulario(texto):
            if pd.isna(texto) or not str(texto).strip():
                return ""

            # Extrai só palavras com 3+ chars (remove pontuação e números)
            palavras = re.findall(r'\b[a-z]{3,}\b', str(texto).lower())

            # Remove stopwords
            palavras_relevantes = [p for p in palavras if p not in stop_words]

            if not palavras_relevantes:
                return ""

            # Conta frequência e retorna as top_n mais frequentes
            contagem = pd.Series(palavras_relevantes).value_counts()
            top_palavras = contagem.head(top_n).index.tolist()

            return " ".join(top_palavras)

        df['vocabulario_falado'] = df['texto_falado_limpo_en'].apply(extrair_vocabulario)

        # Métricas para conferência rápida
        sem_vocabulario = (df['vocabulario_falado'] == "").sum()
        print(" SUCESSO!")
        print(f"    [Vocabulário] : vocabulario_falado (top {top_n} palavras por vídeo)")
        print(f"    Vídeos sem vocabulário : {sem_vocabulario} ({sem_vocabulario/len(df)*100:.1f}%)")

    except Exception as e:
        print(f" Erro na Etapa 13: {e}")

    return df

def maestro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestra a execução sequencial de todas as etapas da Usina de Features.
    Recebe o DataFrame bruto e devolve processado, inserindo as novas features
    na ordem oficial sem alterar o alinhamento das colunas originais da coleta.
    """
    print("\n" + "="*55)
    print("🎬 INICIANDO USINA DE FEATURES (MAESTRO)")
    print("="*55)

    # ── Etapas 1–3: Limpeza e Blindagem ─────────────────────────────────────
    df = remover_colunas_inuteis(df)
    df = tratar_valores_nulos(df)
    df = limpar_erros_transcricao(df)

    # ── Etapa 4: Tração e Metadados Temporais ────────────────────────────────
    # Gera: dia_postagem, hora_postagem, janela_postagem, idade_dias,
    #        velocidade_views, taxa_conversao, taxa_discussao
    df = criar_features_tracao_metadados(df)

    # ── Etapa 5: Faixa de Duração ────────────────────────────────────────────
    # Gera: faixa_duracao
    # Depende de: duracao_segundos (original)
    df = criar_features_faixa_duracao(df)

    # ── Etapa 6: Formato, Direção e Copywriting ──────────────────────────────
    # Gera: pistas_audio, vibe_emojis, palavras_chave
    df = criar_features_formato_copywriting(df)

    # ── Etapa 7: Purificação do Texto Falado ─────────────────────────────────
    # Gera: texto_falado_limpo
    df = purificar_texto_e_calcular_ritmo(df)

    # ── Etapa 8: Unificação de Idioma ────────────────────────────────────────
    # Gera: titulo_en, descricao_en, texto_falado_limpo_en,
    #        palavras_chave_en, pistas_audio_en
    df = unificar_idioma_ingles(df)

    # ── Etapa 9: Visão Computacional (Gemini) ────────────────────────────────
    # Gera: descricao_visual_thumb, texto_thumbnail
    df = extrair_features_thumbnail_gemini(df)

    # ── Etapa 10: Ritmo de Palavras ──────────────────────────────────────────
    # Gera: ritmo_palavras_seg
    # Depende de: texto_falado_limpo_en (etapa 8)
    df = calcular_ritmo_palavras_seg(df)

    # ── Etapa 11: Features de Roteiro ────────────────────────────────────────
    # Todas dependem de: texto_falado_limpo_en (etapa 8)
    # Gera: densidade_roteiro
    df = calcular_densidade_roteiro(df)
    # Gera: tem_repeticao_roteiro
    df = calcular_tem_repeticao_roteiro(df)
    # Gera: gancho_primeira_frase
    df = calcular_gancho_primeira_frase(df)
    # Gera: sentimento_roteiro
    df = calcular_sentimento_roteiro(df)

    # ── Etapa 12: Áudio Dominante ─────────────────────────────────────────────
    # Gera: tipo_audio_dominante
    # Depende de: pistas_audio_en (etapa 8)
    df = calcular_tipo_audio_dominante(df)

    # ── Etapa 13: Estrutura de Blocos ────────────────────────────────────────
    # Gera: estrutura_blocos
    # Depende de: faixa_duracao (etapa 5), ritmo_palavras_seg (etapa 10),
    #             densidade_roteiro, tem_repeticao_roteiro (etapa 11),
    #             tipo_audio_dominante (etapa 12)
    df = calcular_estrutura_blocos(df)

    # ── Etapa 14: Score Viral ────────────────────────────────────────────────
    # Gera: score_viral, label_viral
    # Depende de: estrutura_blocos (etapa 13), ritmo_palavras_seg (etapa 10)
    df = calcular_score_viral(df)

    # ── Etapa 15: Clickbait Score ────────────────────────────────────────────
    # Gera: clickbait_score
    # Depende de: titulo_en, descricao_en (etapa 8),
    #             texto_thumbnail, descricao_visual_thumb (etapa 9)
    df = calcular_clickbait_score(df)

    # ── Etapa 16: Completude SEO ─────────────────────────────────────────────
    # Gera: completude_seo
    # Depende de: titulo_en, descricao_en (etapa 8), tags (original)
    df = calcular_completude_seo(df)

    # ── Etapa 17: Vocabulário Falado ─────────────────────────────────────────
    # Gera: vocabulario_falado
    # Depende de: texto_falado_limpo_en (etapa 8)
    df = criar_vocabulario_falado(df)

    # ── Etapa 18: Reordenação Dinâmica (Proteção de Colunas Originais) ───────
    print("\n" + "-"*55)
    print("ETAPA 18: ALINHAMENTO DE COLUNAS")
    print("-"*55)

    # Lista exata e ordenada APENAS das features geradas por este arquivo .py
    ordem_features_novas = [
        # Metadados Temporais
        'dia_postagem',
        'hora_postagem',
        'janela_postagem',
        'idade_dias',

        # Métricas Calculadas
        'velocidade_views',
        'taxa_conversao',
        'taxa_discussao',

        # Score e Label Viral
        'score_viral',
        'label_viral',

        # Features de Formato
        'faixa_duracao',
        'estrutura_blocos',
        'ritmo_palavras_seg',

        # Features de Roteiro
        'densidade_roteiro',
        'tem_repeticao_roteiro',
        'gancho_primeira_frase',
        'sentimento_roteiro',

        # Áudio e Emoção
        'tipo_audio_dominante',

        # Features de Copywriting e Atração
        'clickbait_score',
        'completude_seo',
        'vibe_emojis',
        'pistas_audio',
        'pistas_audio_en',

        # Semântica e Vocabulário
        'palavras_chave',
        'palavras_chave_en',
        'vocabulario_falado',

        # Conteúdo Traduzido
        'texto_falado_limpo',
        'texto_falado_limpo_en',
        'titulo_en',
        'descricao_en',

        # Thumbnail (Gemini)
        'descricao_visual_thumb',
        'texto_thumbnail',
    ]

    # Identifica o que sobrou das colunas originais presentes no DF, mantendo a ordem exata delas
    colunas_originais_preservadas = [col for col in df.columns if col not in ordem_features_novas]

    # Filtra apenas as colunas novas que realmente foram calculadas com sucesso (evita KeyError)
    features_calculadas_presentes = [col for col in ordem_features_novas if col in df.columns]

    # Junta as originais intactas com as novas na ordem estipulada
    df = df[colunas_originais_preservadas + features_calculadas_presentes]

    print(" Usina processada com sucesso!")
    print(f" Colunas da coleta preservadas : {len(colunas_originais_preservadas)}")
    print(f" Novas features acopladas      : {len(features_calculadas_presentes)}")
    print(f" Total final de colunas        : {df.shape[1]}")
    print("="*55 + "\n")

    return df

