"""
00_ajuste.py
============
Recria todas as features engineered do CSV prata do zero,
espelhando fielmente a lógica do processador_features.py (o pai).

CSV de entrada/saída: dataset_youtube_processado_modulo2.csv
(mesmo diretório do script)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUNAS PRESERVADAS (custo alto de regerar via API — não tocadas):
    descricao_visual_thumb
    texto_thumbnail

COLUNAS ORIGINAIS (vindas da coleta — nunca dropadas):
    video_id, titulo, descricao, tags, texto_falado, canal_id,
    canal_nome, visualizacoes, curtidas, comentarios, data_publicacao,
    duracao_segundos, tem_legenda_nativa, conteudo_licenciado,
    feito_para_criancas, categoria_id, idioma_audio_default,
    idioma_texto_default, topicos_wikipedia, thumb_maxres, nicho

Todas as outras colunas são DROPADAS e recriadas na ordem abaixo,
que é exatamente a mesma do processador_features.py (etapa 18).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORDEM DE EXECUÇÃO E DEPENDÊNCIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FASE 1 — sem dependências externas
    TEMPORAL  : dia_postagem, hora_postagem, janela_postagem, idade_dias
    TRAÇÃO    : velocidade_views, taxa_conversao, taxa_discussao
    FORMATO   : faixa_duracao

  FASE 2 — limpeza de texto (base para _en)
    TEXTO     : texto_falado_limpo, palavras_chave, pistas_audio

  FASE 3 — tradução (alimenta features downstream)
    _EN       : texto_falado_limpo_en, titulo_en, descricao_en,
                palavras_chave_en, pistas_audio_en

  FASE 4 — features que consomem _en
    ROTEIRO   : ritmo_palavras_seg, densidade_roteiro,
                tem_repeticao_roteiro, gancho_primeira_frase,
                sentimento_roteiro
    ÁUDIO     : tipo_audio_dominante
    EMOÇÃO    : vibe_emojis
    FORMATO   : estrutura_blocos
    TRAÇÃO    : score_viral, label_viral
    SEO       : clickbait_score, completude_seo
    SEMÂNTICA : vocabulario_falado

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import ast
import sys
import os
import re
import textwrap

import emoji
import nltk
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibrar_limiares import carregar_limiares

# ─────────────────────────────────────────────────────────────
# CAMINHO DO CSV
# ─────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset_youtube_processado_modulo2.csv")

# ─────────────────────────────────────────────────────────────
# COLUNAS PRESERVADAS (não serão dropadas nem recriadas)
# ─────────────────────────────────────────────────────────────

COLUNAS_PRESERVADAS = [
    "descricao_visual_thumb",
    "texto_thumbnail",
]

# ─────────────────────────────────────────────────────────────
# COLUNAS ORIGINAIS (vindas da coleta — nunca dropadas)
# ─────────────────────────────────────────────────────────────

COLUNAS_ORIGINAIS = [
    "video_id",
    "titulo",
    "descricao",
    "tags",
    "texto_falado",
    "canal_id",
    "canal_nome",
    "visualizacoes",
    "curtidas",
    "comentarios",
    "data_publicacao",
    "duracao_segundos",
    "tem_legenda_nativa",
    "conteudo_licenciado",
    "feito_para_criancas",
    "categoria_id",
    "idioma_audio_default",
    "idioma_texto_default",
    "topicos_wikipedia",
    "thumb_maxres",
    "nicho",
]

# Ordem final espelhando exatamente a etapa 18 do processador_features.py
ORDEM_FEATURES_CRIADAS = [
    # Metadados Temporais
    "dia_postagem",
    "hora_postagem",
    "janela_postagem",
    "idade_dias",

    # Métricas Calculadas
    "velocidade_views",
    "taxa_conversao",
    "taxa_discussao",

    # Score e Label Viral
    "score_viral",
    "label_viral",

    # Features de Formato
    "faixa_duracao",
    "estrutura_blocos",
    "ritmo_palavras_seg",

    # Features de Roteiro
    "densidade_roteiro",
    "tem_repeticao_roteiro",
    "gancho_primeira_frase",
    "sentimento_roteiro",

    # Áudio e Emoção
    "tipo_audio_dominante",

    # Features de Copywriting e Atração
    "clickbait_score",
    "completude_seo",
    "vibe_emojis",
    "pistas_audio",
    "pistas_audio_en",

    # Semântica e Vocabulário
    "palavras_chave",
    "palavras_chave_en",
    "vocabulario_falado",

    # Conteúdo Traduzido
    "texto_falado_limpo",
    "texto_falado_limpo_en",
    "titulo_en",
    "descricao_en",

    # Thumbnail (Gemini — preservadas, mas entram na ordem aqui)
    "descricao_visual_thumb",
    "texto_thumbnail",
]

# Mapeamento de categoria_id para nome legível (igual ao pai)
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
    "41": "Thriller", "42": "Shorts", "43": "Shows", "44": "Trailers",
}


# ─────────────────────────────────────────────────────────────
# DROP — apaga todas as criadas exceto as preservadas
# ─────────────────────────────────────────────────────────────

def dropar_colunas_criadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove todas as colunas que não são originais e não estão
    na lista de preservadas. Garante um slate limpo antes de
    recriar tudo na ordem correta.
    """
    manter = set(COLUNAS_ORIGINAIS) | set(COLUNAS_PRESERVADAS)
    dropar = [col for col in df.columns if col not in manter]
    df = df.drop(columns=dropar)
    print(f"  [DROP] {len(dropar)} colunas removidas.")
    if dropar:
        print(f"         {dropar}")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 1 — BLOCO TEMPORAL
# Fonte: criar_features_tracao_metadados() do pai
# ─────────────────────────────────────────────────────────────

def calcular_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    dia_postagem  — nome do dia da semana em lowercase.
    hora_postagem — hora inteira (0-23).
    janela_postagem — madrugada (0-5h) | manha (6-11h) | tarde (12-17h) | noite (18-23h)
    idade_dias    — dias desde data_publicacao até hoje. Mínimo 1, teto 365.

    .log: sexta domina 21x | quarta é a menor 8x
    .log: 80% das postagens entre 3h e 14h — madrugada e manhã dominam
    .log: Q1=35d | Q2=83d | Q3=103d | Q4=119d
    """
    df["data_publicacao_dt"] = pd.to_datetime(df["data_publicacao"], utc=True)

    df["dia_postagem"]  = df["data_publicacao_dt"].dt.day_name().str.lower()
    df["hora_postagem"] = df["data_publicacao_dt"].dt.hour

    hora = df["hora_postagem"]
    condicoes = [hora <= 5, hora <= 11, hora <= 17]
    escolhas  = ["madrugada", "manha", "tarde"]
    df["janela_postagem"] = np.select(condicoes, escolhas, default="noite")

    data_atual  = datetime.now(timezone.utc)
    idade_bruta = (data_atual - df["data_publicacao_dt"]).dt.days
    df["idade_dias"] = np.clip(idade_bruta, 1, 365)

    df = df.drop(columns=["data_publicacao_dt"], errors="ignore")

    print("  [OK] dia_postagem | hora_postagem | janela_postagem | idade_dias")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 1 — BLOCO TRAÇÃO
# Fonte: criar_features_tracao_metadados() do pai
# ─────────────────────────────────────────────────────────────

def calcular_tracao_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    velocidade_views — visualizacoes / idade_dias.
    taxa_conversao   — curtidas / visualizacoes * 100.
    taxa_discussao   — comentarios / visualizacoes * 100.

    .log: velocidade mediana 353.194 | máximo 2.210.149
    .log: conversao mediana 1.39% | pico 4.73%
    .log: discussao mediana 0.0032% | pico 0.0257%

    Depende de: idade_dias (calcular_temporal).
    """
    for col in ["visualizacoes", "curtidas", "comentarios"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["velocidade_views"] = df["visualizacoes"] / df["idade_dias"]

    df["taxa_conversao"] = np.where(
        df["visualizacoes"] > 0,
        (df["curtidas"]    / df["visualizacoes"]) * 100,
        0,
    )
    df["taxa_discussao"] = np.where(
        df["visualizacoes"] > 0,
        (df["comentarios"] / df["visualizacoes"]) * 100,
        0,
    )

    print("  [OK] velocidade_views | taxa_conversao | taxa_discussao")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 1 — BLOCO FORMATO
# Fonte: criar_features_faixa_duracao() do pai
# ─────────────────────────────────────────────────────────────

def calcular_faixa_duracao(df: pd.DataFrame) -> pd.DataFrame:
    """
    faixa_duracao — categórica baseada nos limites técnicos do YouTube Shorts.
      ultra_curto  : <= 30s
      short_padrao : 31-60s
      short_longo  : > 60s

    .log: mediana 52s | max 181s | 59% <= 60s
    Consumido por: estrutura_blocos (FASE 4).
    """
    condicoes = [
        df["duracao_segundos"] <= 30,
        (df["duracao_segundos"] > 30) & (df["duracao_segundos"] <= 60),
        df["duracao_segundos"] > 60,
    ]
    escolhas = ["ultra_curto", "short_padrao", "short_longo"]
    df["faixa_duracao"] = np.select(condicoes, escolhas, default="desconhecido")

    print("  [OK] faixa_duracao")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 2 — TEXTO LIMPO
# ─────────────────────────────────────────────────────────────

def calcular_texto_falado_limpo(df: pd.DataFrame) -> pd.DataFrame:
    """
    texto_falado_limpo — texto_falado sem tags ASR, normalizado em lowercase.
    Remove: [tag], (tag), ♪
    Fonte: purificar_texto_e_calcular_ritmo() do pai.

    .log: tags mais comuns: [हंसी], [संगीत], [laughter], [音楽]
    """
    def purificar(texto):
        if pd.isna(texto):
            return ""
        texto_sem_tags = re.sub(r"\[.*?\]|\(.*?\)|♪", " ", str(texto))
        return re.sub(r"\s+", " ", texto_sem_tags).strip().lower()

    df["texto_falado"] = df["texto_falado"].fillna("")
    # Blindagem: mata a anomalia do Pandas que converte nulo para a string "nan"
    mascara = df["texto_falado"].astype(str).str.strip().str.lower() == "nan"
    df.loc[mascara, "texto_falado"] = ""

    df["texto_falado_limpo"] = df["texto_falado"].apply(purificar)

    print("  [OK] texto_falado_limpo")
    return df


def calcular_pistas_audio(df: pd.DataFrame) -> pd.DataFrame:
    """
    pistas_audio — tags ASR do texto_falado ORIGINAL (não do limpo).
    Captura: [tag], (tag), ♪. Remove duplicatas, preserva ordem.
    Fonte: criar_features_formato_copywriting() do pai.

    .log: [संगीत] 16x | [हंसी] 16x | [laughter] 3x | [screaming] 2x
    Par com pistas_audio_en (FASE 3).
    """
    def extrair(texto):
        if pd.isna(texto):
            return ""
        matches = re.findall(r"\[.*?\]|\(.*?\)|♪", str(texto))
        return " ".join(list(dict.fromkeys(matches))) if matches else ""

    df["pistas_audio"] = df["texto_falado"].apply(extrair).str.strip()

    print("  [OK] pistas_audio")
    return df


def calcular_palavras_chave(df: pd.DataFrame) -> pd.DataFrame:
    """
    palavras_chave — unificação de topicos_wikipedia + categoria_id
    + hashtags do titulo e descricao + tags em string limpa, idioma nativo.
    Fonte: criar_features_formato_copywriting() do pai.

    Par com palavras_chave_en (FASE 3).
    """
    from urllib.parse import unquote

    def construir(row):
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

        titulo    = str(row.get("titulo", ""))
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

        return ", ".join(sorted(palavras_limpas))

    df["palavras_chave"] = df.apply(construir, axis=1)

    print("  [OK] palavras_chave")
    return df


def calcular_vibe_emojis(df: pd.DataFrame) -> pd.DataFrame:
    """
    vibe_emojis — string com os emojis únicos do titulo + descricao, ordenados.
    Deduplicação com set() para evitar inflação de emojis que aparecem
    tanto no título quanto na descrição.
    Fonte: criar_features_formato_copywriting() do pai.

    .log: humor domina — 😂 35x | 🤣 28x | 😜 40x | 🤪 40x
    """
    def extrair_emojis(texto):
        texto_str = str(texto)
        if texto_str == "nan" or not texto_str.strip():
            return set()
        return set(c["emoji"] for c in emoji.emoji_list(texto_str))

    df["descricao"] = df["descricao"].fillna("")
    mascara = df["descricao"].astype(str).str.strip().str.lower() == "nan"
    df.loc[mascara, "descricao"] = ""

    df["vibe_emojis"] = df.apply(
        lambda row: "".join(sorted(
            extrair_emojis(row["titulo"]) | extrair_emojis(row["descricao"])
        )),
        axis=1,
    )

    print("  [OK] vibe_emojis")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 3 — TRADUÇÕES _EN
# Fonte: unificar_idioma_ingles() do pai
# ─────────────────────────────────────────────────────────────

def calcular_traducoes_en(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduz 5 colunas para inglês usando GoogleTranslator com chunking.
    Gera: texto_falado_limpo_en, titulo_en, descricao_en,
          palavras_chave_en, pistas_audio_en

    .log: 86% do texto falado em devanagari — sem tradução NLP não funciona.

    Depende de: texto_falado_limpo, palavras_chave, pistas_audio (FASE 2).
    """
    tqdm.pandas()
    tradutor = GoogleTranslator(source="auto", target="en")

    def traduzir(texto):
        if pd.isna(texto) or str(texto).strip() == "":
            return ""
        texto_str = str(texto)
        if len(texto_str) < 4900:
            try:
                return tradutor.translate(texto_str)
            except Exception:
                return texto_str
        # Chunking para textos grandes
        pedacos = textwrap.wrap(texto_str, width=4500, break_long_words=False)
        resultado = ""
        for pedaco in pedacos:
            try:
                resultado += tradutor.translate(pedaco) + " "
            except Exception:
                resultado += pedaco + " "
        return resultado.strip()

    alvos = [
        ("titulo",              "titulo_en",              False),
        ("descricao",           "descricao_en",           False),
        ("texto_falado_limpo",  "texto_falado_limpo_en",  True),
        ("palavras_chave",      "palavras_chave_en",      True),
        ("pistas_audio",        "pistas_audio_en",        True),
    ]

    for i, (origem, destino, lowercase) in enumerate(alvos, 1):
        print(f"  {i}/5: Traduzindo {origem} → {destino}...")
        if origem in df.columns:
            serie = df[origem].progress_apply(traduzir)
            df[destino] = serie.str.lower() if lowercase else serie

    print("  [OK] titulo_en | descricao_en | texto_falado_limpo_en | palavras_chave_en | pistas_audio_en")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO ROTEIRO
# ─────────────────────────────────────────────────────────────

def calcular_ritmo_palavras_seg(df: pd.DataFrame) -> pd.DataFrame:
    """
    ritmo_palavras_seg — palavras(texto_falado_limpo_en) / duracao_segundos.
    Usa _en para tokenização confiável — devanagari não tokeniza
    por espaço da mesma forma que o inglês, distorcendo o ritmo.
    Fonte: calcular_ritmo_palavras_seg() do pai.

    .log: mediana 2.24 pal/seg | 6% de vídeos com ritmo zero
    Consumido por: estrutura_blocos, score_viral.
    Depende de: texto_falado_limpo_en (FASE 3).
    """
    contagem = df["texto_falado_limpo_en"].fillna("").apply(lambda x: len(str(x).split()))
    df["ritmo_palavras_seg"] = np.where(
        df["duracao_segundos"] > 0,
        contagem / df["duracao_segundos"],
        0.0,
    )
    print("  [OK] ritmo_palavras_seg")
    return df


def calcular_densidade_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    densidade_roteiro — total de palavras no texto_falado_limpo_en.
    Fonte: calcular_densidade_roteiro() do pai.

    .log: min 2 | max 647 | mediana 97 palavras
    Consumido por: estrutura_blocos.
    Depende de: texto_falado_limpo_en (FASE 3).
    """
    df["densidade_roteiro"] = (
        df["texto_falado_limpo_en"].fillna("").apply(lambda x: len(str(x).split()))
    )
    print("  [OK] densidade_roteiro")
    return df


def calcular_tem_repeticao_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    tem_repeticao_roteiro — True se alguma palavra de conteúdo aparece 5x+
    no texto_falado_limpo_en (após remoção de stopwords inglês).
    Threshold 5x captura hook intencional sem confundir com recorrência natural.
    Fonte: calcular_tem_repeticao_roteiro() do pai.

    .log: 31.9% com repetição — técnica de hook em Shorts
    Consumido por: estrutura_blocos.
    Depende de: texto_falado_limpo_en (FASE 3).
    """
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    def tem_repeticao(t):
        if pd.isna(t) or not str(t).strip():
            return False
        palavras    = re.findall(r"\b[a-z]{3,}\b", str(t).lower())
        relevantes  = [p for p in palavras if p not in stop_words]
        if not relevantes:
            return False
        return bool(pd.Series(relevantes).value_counts().iloc[0] >= 5)

    df["tem_repeticao_roteiro"] = df["texto_falado_limpo_en"].apply(tem_repeticao)
    print("  [OK] tem_repeticao_roteiro")
    return df


def calcular_gancho_primeira_frase(df: pd.DataFrame) -> pd.DataFrame:
    """
    gancho_primeira_frase — primeiras 15 palavras de texto_falado_limpo_en.
    Isola o hook de abertura — em Shorts os primeiros ~3s determinam retenção.
    Fonte: calcular_gancho_primeira_frase() do pai.

    Depende de: texto_falado_limpo_en (FASE 3).
    """
    def gancho(t):
        if pd.isna(t) or not str(t).strip():
            return ""
        return " ".join(str(t).split()[:15])

    df["gancho_primeira_frase"] = df["texto_falado_limpo_en"].apply(gancho)
    print("  [OK] gancho_primeira_frase")
    return df


def calcular_sentimento_roteiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    sentimento_roteiro — tom emocional do roteiro: positivo | negativo | neutro.
    Usa nltk VADER sobre texto_falado_limpo_en.
    Limiares padrão VADER: compound >= 0.05 → positivo | <= -0.05 → negativo.
    Fonte: calcular_sentimento_roteiro() do pai.

    Depende de: texto_falado_limpo_en (FASE 3).
    """
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

    def sentimento(t):
        if pd.isna(t) or not str(t).strip():
            return "neutro"
        compound = sia.polarity_scores(str(t))["compound"]
        if compound >= 0.05:
            return "positivo"
        if compound <= -0.05:
            return "negativo"
        return "neutro"

    df["sentimento_roteiro"] = df["texto_falado_limpo_en"].apply(sentimento)
    print("  [OK] sentimento_roteiro")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO ÁUDIO
# ─────────────────────────────────────────────────────────────

def calcular_tipo_audio_dominante(df: pd.DataFrame) -> pd.DataFrame:
    """
    tipo_audio_dominante — tag de áudio mais frequente em pistas_audio_en.
    Retorna a tag mais frequente como string limpa, ou "" quando ausente.
    Fonte: calcular_tipo_audio_dominante() do pai.

    .log: 75% são "" — o vazio já é informativo
    Consumido por: estrutura_blocos.
    Depende de: pistas_audio_en (FASE 3).
    """
    def dominante(t):
        if pd.isna(t) or not str(t).strip():
            return ""
        tags = re.findall(r"\[.*?\]|\(.*?\)|♪", str(t))
        if not tags:
            return ""
        return pd.Series(tags).value_counts().index[0]

    df["tipo_audio_dominante"] = df["pistas_audio_en"].apply(dominante)
    print("  [OK] tipo_audio_dominante")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO FORMATO
# ─────────────────────────────────────────────────────────────


def calcular_estrutura_blocos(df: pd.DataFrame) -> pd.DataFrame:
    """
    estrutura_blocos — formato narrativo real do vídeo via score de compatibilidade.
    5 sinais combinados; label com maior score vence.
    Limiares calculados por quartis do dataset — universal.
    Labels: impacto_rapido | esquete | esquete_com_hook | narrativa
    Fonte: calcular_estrutura_blocos() do pai.

    Depende de: faixa_duracao (FASE 1), ritmo_palavras_seg,
                densidade_roteiro, tem_repeticao_roteiro (FASE 4 | ROTEIRO),
                tipo_audio_dominante (FASE 4 | ÁUDIO).
    """
    def score(row, q1_den, q3_den, q1_rit, q3_rit):
        faixa     = row["faixa_duracao"]
        ritmo     = row["ritmo_palavras_seg"]
        density   = row["densidade_roteiro"]
        repeticao = row["tem_repeticao_roteiro"]
        audio     = str(row["tipo_audio_dominante"]).strip()

        scores = {
            "impacto_rapido"  : 0.0,
            "esquete"         : 0.0,
            "esquete_com_hook": 0.0,
            "narrativa"       : 0.0,
        }

        if faixa == "ultra_curto":
            scores["impacto_rapido"]   += 0.25
        elif faixa == "short_padrao":
            scores["esquete"]          += 0.15
            scores["esquete_com_hook"] += 0.15
        else:
            scores["narrativa"]        += 0.25

        if ritmo >= q3_rit:
            scores["impacto_rapido"]   += 0.20
            scores["esquete_com_hook"] += 0.10
        elif ritmo <= q1_rit:
            scores["narrativa"]        += 0.20
            scores["esquete"]          += 0.10
        else:
            scores["esquete"]          += 0.15
            scores["esquete_com_hook"] += 0.15

        if density <= q1_den:
            scores["impacto_rapido"]   += 0.20
        elif density >= q3_den:
            scores["narrativa"]        += 0.20
        else:
            scores["esquete"]          += 0.15
            scores["esquete_com_hook"] += 0.15

        if repeticao:
            scores["esquete_com_hook"] += 0.20
            scores["impacto_rapido"]   += 0.05
        else:
            scores["esquete"]          += 0.10
            scores["narrativa"]        += 0.10

        if audio:
            scores["impacto_rapido"]   += 0.15
            scores["esquete_com_hook"] += 0.05
        else:
            scores["narrativa"]        += 0.10
            scores["esquete"]          += 0.10

        return max(scores, key=scores.get)

    partes = []
    for nicho, df_nicho in df.groupby("nicho"):
        limiares = carregar_limiares()[nicho]["estrutura_blocos"]
        q1_den = limiares["densidade_roteiro"]["q1"]
        q3_den = limiares["densidade_roteiro"]["q3"]
        q1_rit = limiares["ritmo_palavras_seg"]["q1"]
        q3_rit = limiares["ritmo_palavras_seg"]["q3"]

        df_nicho = df_nicho.copy()
        df_nicho["estrutura_blocos"] = df_nicho.apply(
            lambda row: score(row, q1_den, q3_den, q1_rit, q3_rit), axis=1
        )

        print(f"  [OK] estrutura_blocos | nicho={nicho}")
        print(f"       limiares: densidade Q1={q1_den:.0f} Q3={q3_den:.0f} | "
              f"ritmo Q1={q1_rit:.2f} Q3={q3_rit:.2f}")
        print(f"       distribuição: {df_nicho['estrutura_blocos'].value_counts().to_dict()}")

        partes.append(df_nicho)

    df = pd.concat(partes).sort_index()
    return df

# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO TRAÇÃO FINAL
# Fonte: calcular_score_viral() do pai
# ─────────────────────────────────────────────────────────────


def calcular_score_viral(df: pd.DataFrame) -> pd.DataFrame:
    """
    score_viral  — score 0-1 normalizado via Min-Max.
    label_viral  — quartil categórico: frio | aquecido | viral | super_viral.

    Pesos fixos calibrados para Shorts:
      velocidade_views : 0.60 — velocidade de views define distribuição do algoritmo
      taxa_conversao   : 0.30 — curtidas são o principal sinal de qualidade em Shorts
      taxa_discussao   : 0.10 — comentários são raros mas informativos

    Depende de: velocidade_views, taxa_conversao, taxa_discussao (FASE 1).
    """
    def minmax(s, mn, mx):
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def classificar_label(score, cortes):
        if score <= cortes["q25"]:
            return "frio"
        elif score <= cortes["q50"]:
            return "aquecido"
        elif score <= cortes["q75"]:
            return "viral"
        else:
            return "super_viral"

    peso_views, peso_conversao, peso_discussao = 0.60, 0.30, 0.10

    partes = []
    for nicho, df_nicho in df.groupby("nicho"):
        limiares_nicho = carregar_limiares()[nicho]
        limiares_score = limiares_nicho["score_viral"]
        cortes = limiares_nicho["label_viral"]["cortes_score_viral"]

        df_nicho = df_nicho.copy()

        views_norm     = minmax(df_nicho["velocidade_views"], limiares_score["velocidade_views"]["min"], limiares_score["velocidade_views"]["max"])
        conversao_norm = minmax(df_nicho["taxa_conversao"], limiares_score["taxa_conversao"]["min"], limiares_score["taxa_conversao"]["max"])
        discussao_norm = minmax(df_nicho["taxa_discussao"], limiares_score["taxa_discussao"]["min"], limiares_score["taxa_discussao"]["max"])

        df_nicho["score_viral"] = (
            (views_norm     * peso_views)     +
            (conversao_norm * peso_conversao) +
            (discussao_norm * peso_discussao)
        )

        df_nicho["label_viral"] = df_nicho["score_viral"].apply(lambda score: classificar_label(score, cortes))

        print(f"  [OK] score_viral | label_viral | nicho={nicho}")
        print(f"       pesos: velocidade={peso_views:.0%} | conversao={peso_conversao:.0%} | "
              f"discussao={peso_discussao:.0%}")

        partes.append(df_nicho)

    df = pd.concat(partes).sort_index()
    return df


# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO SEO
# ─────────────────────────────────────────────────────────────

def calcular_clickbait_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    clickbait_score — score 0-1 normalizado. 3 pilares com peso igual:
      A) Título     : emojis + hashtags + ratio CAPS/!? + score de comprimento
      B) Thumbnail  : presença e tamanho do texto na capa + riqueza visual
      C) Descrição  : hashtags + emojis + força da primeira linha
    Fonte: calcular_clickbait_score() do pai.

    Depende de: titulo_en, descricao_en (FASE 3),
                texto_thumbnail, descricao_visual_thumb (PRESERVADAS).
    """
    def minmax(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def score_titulo(texto):
        if pd.isna(texto) or not str(texto).strip():
            return 0.0
        t = str(texto)
        qtd_emojis   = len(emoji.emoji_list(t))
        qtd_hashtags = len(re.findall(r"#\w+", t))
        chars_alpha  = sum(1 for c in t if c.isalpha())
        caps         = sum(1 for c in t if c.isupper())
        pontuacao    = len(re.findall(r"[!?]", t))
        ratio_caps   = (caps + pontuacao) / (chars_alpha + pontuacao) if (chars_alpha + pontuacao) > 0 else 0.0
        comprimento  = len(t)
        score_comp   = 1.0 if 40 <= comprimento <= 70 else max(0.0, 1.0 - abs(comprimento - 55) / 55)
        return qtd_emojis + qtd_hashtags + ratio_caps + score_comp

    def score_thumbnail(texto_thumb, desc_visual):
        pontos = 0.0
        t = str(texto_thumb) if not pd.isna(texto_thumb) else ""
        if t.strip():
            pontos += 1.0
            pontos += max(0.0, 1.0 - max(0, len(t) - 40) / 40)
        d = str(desc_visual) if not pd.isna(desc_visual) else ""
        if d.strip():
            pontos += min(1.0, len(d) / 100)
        return pontos

    def score_descricao(texto):
        if pd.isna(texto) or not str(texto).strip():
            return 0.0
        t = str(texto)
        qtd_hashtags     = len(re.findall(r"#\w+", t))
        qtd_emojis       = len(emoji.emoji_list(t))
        primeira_linha   = t[:100].strip()
        score_prim_linha = min(1.0, len(primeira_linha) / 100)
        return qtd_hashtags + qtd_emojis + score_prim_linha

    raw_titulo = df["titulo_en"].apply(score_titulo)
    raw_thumb  = df.apply(
        lambda r: score_thumbnail(r["texto_thumbnail"], r["descricao_visual_thumb"]), axis=1
    )
    raw_desc   = df["descricao_en"].apply(score_descricao)

    df["clickbait_score"] = (minmax(raw_titulo) + minmax(raw_thumb) + minmax(raw_desc)) / 3.0

    print("  [OK] clickbait_score")
    return df


def calcular_completude_seo(df: pd.DataFrame) -> pd.DataFrame:
    """
    completude_seo — score contínuo de intensidade de SEO e copywriting.
    Sinais: descricao preenchida, tags, hashtags, menções @, pontuação !?, emojis.
    Fonte: calcular_completude_seo() do pai.

    Depende de: titulo_en, descricao_en (FASE 3), tags (original).
    """
    def score(row):
        pontos = 0.0
        titulo = str(row.get("titulo_en", ""))
        desc   = str(row.get("descricao_en", ""))

        if desc.strip() and desc.strip().lower() != "nan":
            pontos += 1

        tags = row.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = ast.literal_eval(tags)
            except (ValueError, SyntaxError):
                tags = []
        if isinstance(tags, list) and len(tags) > 0:
            pontos += 1

        texto_completo = titulo + " " + desc
        pontos += len(re.findall(r"#\w+", texto_completo))
        pontos += len(re.findall(r"@\w+", texto_completo))
        pontos += len(re.findall(r"[!?]", texto_completo))
        pontos += len(emoji.emoji_list(texto_completo))

        return pontos

    df["completude_seo"] = df.apply(score, axis=1)

    print("  [OK] completude_seo")
    return df


# ─────────────────────────────────────────────────────────────
# FASE 4 — BLOCO SEMÂNTICO
# Fonte: criar_vocabulario_falado() do pai
# ─────────────────────────────────────────────────────────────

def calcular_vocabulario_falado(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    vocabulario_falado — top_n palavras mais frequentes de texto_falado_limpo_en,
    após remoção de stopwords inglês. String separada por espaços.
    Complementa palavras_chave_en com o vocabulário real da fala.
    Fonte: criar_vocabulario_falado() do pai.

    .log: mediana 97 palavras | 6% de vídeos sem roteiro
    Depende de: texto_falado_limpo_en (FASE 3).
    """
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    def extrair_vocab(texto):
        if pd.isna(texto) or not str(texto).strip():
            return ""
        palavras    = re.findall(r"\b[a-z]{3,}\b", str(texto).lower())
        relevantes  = [p for p in palavras if p not in stop_words]
        if not relevantes:
            return ""
        contagem    = pd.Series(relevantes).value_counts()
        return " ".join(contagem.head(top_n).index.tolist())

    df["vocabulario_falado"] = df["texto_falado_limpo_en"].apply(extrair_vocab)

    print("  [OK] vocabulario_falado")
    return df


# ─────────────────────────────────────────────────────────────
# REORDENAÇÃO FINAL
# Espelha exatamente a etapa 18 do processador_features.py
# ─────────────────────────────────────────────────────────────

def reordenar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena o DataFrame na ordem final definida em ORDEM_FEATURES_CRIADAS,
    precedida pelas colunas originais da coleta.
    Colunas presentes no DF mas ausentes das listas vão para o fim (segurança).
    """
    originais_presentes = [c for c in COLUNAS_ORIGINAIS if c in df.columns]
    features_presentes  = [c for c in ORDEM_FEATURES_CRIADAS
                           if c in df.columns and c not in COLUNAS_ORIGINAIS]
    extras = [c for c in df.columns
              if c not in originais_presentes and c not in features_presentes]

    df = df[originais_presentes + features_presentes + extras]
    print(f"  [ORDER] {len(originais_presentes)} originais | "
          f"{len(features_presentes)} features | {len(extras)} extras ao fim")
    return df


# ─────────────────────────────────────────────────────────────
# ORQUESTRADOR
# ─────────────────────────────────────────────────────────────

def processar(df: pd.DataFrame) -> pd.DataFrame:
    """Executa todas as fases em sequência respeitando as dependências."""

    print("\n[DROP] Removendo colunas criadas (exceto preservadas)...")
    df = dropar_colunas_criadas(df)

    # ── FASE 1 ────────────────────────────────────────────────
    print("\n[FASE 1 | TEMPORAL + TRAÇÃO + FORMATO]")
    df = calcular_temporal(df)
    df = calcular_tracao_base(df)
    df = calcular_faixa_duracao(df)

    # ── FASE 2 ────────────────────────────────────────────────
    print("\n[FASE 2 | TEXTO LIMPO]")
    df = calcular_texto_falado_limpo(df)
    df = calcular_pistas_audio(df)
    df = calcular_palavras_chave(df)
    df = calcular_vibe_emojis(df)

    # ── FASE 3 ────────────────────────────────────────────────
    print("\n[FASE 3 | TRADUÇÕES _EN]")
    df = calcular_traducoes_en(df)

    # ── FASE 4 ────────────────────────────────────────────────
    print("\n[FASE 4 | ROTEIRO]")
    df = calcular_ritmo_palavras_seg(df)
    df = calcular_densidade_roteiro(df)
    df = calcular_tem_repeticao_roteiro(df)
    df = calcular_gancho_primeira_frase(df)
    df = calcular_sentimento_roteiro(df)

    print("\n[FASE 4 | ÁUDIO]")
    df = calcular_tipo_audio_dominante(df)

    print("\n[FASE 4 | FORMATO REAL]")
    df = calcular_estrutura_blocos(df)

    print("\n[FASE 4 | TRAÇÃO FINAL]")
    df = calcular_score_viral(df)

    print("\n[FASE 4 | SEO]")
    df = calcular_clickbait_score(df)
    df = calcular_completude_seo(df)

    print("\n[FASE 4 | SEMÂNTICA]")
    df = calcular_vocabulario_falado(df)

    print("\n[ORDEM FINAL]")
    df = reordenar_colunas(df)

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Carregando CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if df is None or df.empty:
        print("Erro: DataFrame vazio. Abortando.")
        return

    print(f"Linhas: {len(df):,}  |  Colunas: {len(df.columns)}")
    df = processar(df)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSalvo em: {CSV_PATH}")


if __name__ == "__main__":
    main()