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
    """
    # Lista exata das colunas que decidimos remover do escopo global
    colunas_para_remover = [
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
        'projecao'
    ]

    # Realiza o drop das colunas no dataframe principal
    # O errors='ignore' garante que o código não quebre caso uma dessas colunas já não exista
    df = df.drop(columns=colunas_para_remover, errors='ignore')

    print(f"✅ Etapa 1 Concluída: {len(colunas_para_remover)} colunas inúteis foram removidas com sucesso.")
    print(f"Total de colunas restantes: {df.shape[1]}")
    
    return df

def tratar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes (NaN) nas colunas de texto e imagens.
    Garante que as operações futuras não quebrem por falta de dados.
    """
    # 1. Tratamento da coluna 'descricao': substitui NaN por string vazia ""
    if 'descricao' in df.columns:
        df['descricao'] = df['descricao'].fillna("")

    # 2. Tratamento da coluna 'thumb_maxres': substitui NaN pelo valor da 'thumb_default' da mesma linha
    if 'thumb_maxres' in df.columns and 'thumb_default' in df.columns:
        df['thumb_maxres'] = df['thumb_maxres'].fillna(df['thumb_default'])

    print("✅ Etapa 2 Concluída: Valores nulos tratados com segurança.")
    
    # Adicionei um if rápido nos prints apenas para evitar erro caso a coluna não exista no df
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

        print(f"✅ Etapa 3 Concluída: {linhas_com_erro} linhas com falhas de raspagem no 'texto_falado' foram esvaziadas.")
    else:
        print("⚠️ A coluna 'texto_falado' não foi encontrada no DataFrame.")
        
    return df

def criar_features_tracao_metadados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa Engenharia de Tração e Metadados temporais.
    Calcula idade, velocidade de views, engajamento e o Score Viral.
    """
    print("Iniciando Etapa 4: Processando Engenharia de Tração e Metadados...")

    try:
        # 1. Metadados Temporais (Dia e Hora)
        # Convertendo a string de data para datetime com fuso horário UTC (padrão da API do YouTube)
        df['data_publicacao_dt'] = pd.to_datetime(df['data_publicacao'], utc=True)

        # Extraindo o dia da semana em texto e a hora inteira
        df['dia_postagem'] = df['data_publicacao_dt'].dt.day_name()
        df['hora_postagem'] = df['data_publicacao_dt'].dt.hour

        # 2. Engenharia de Tração (Idade e Velocidade)
        # Pega o exato momento de agora em UTC para calcular quantos dias de vida o vídeo tem
        data_atual = datetime.now(timezone.utc)
        df['idade_dias'] = (data_atual - df['data_publicacao_dt']).dt.days

        # Prevenção Crítica: Se um vídeo foi postado hoje (0 dias), forçamos para 1 dia para evitar erro de divisão por zero (ZeroDivisionError)
        df['idade_dias'] = np.where(df['idade_dias'] <= 0, 1, df['idade_dias'])

        # Métrica Mestra: Qual é o tráfego médio diário desse vídeo?
        df['velocidade_views'] = df['visualizacoes'] / df['idade_dias']

        # 3. Engenharia de Engajamento (%)
        # Usamos np.where para evitar divisão por zero caso o vídeo não tenha nenhuma visualização
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

        # 4. Score Viral (A seleção natural do banco)
        # Fórmula: A tração (velocidade) ganha um bônus multiplicador baseado no engajamento real
        multiplicador_engajamento = 1 + (df['taxa_conversao'] / 100) + (df['taxa_discussao'] / 100)
        df['score_viral'] = df['velocidade_views'] * multiplicador_engajamento

        # Limpeza: Deletamos a coluna datetime temporária para manter o dataset otimizado para o Módulo 3
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
    Processa Engenharia de Formato, Pacing, Direção e Copywriting.
    Extrai pistas de áudio, ritmo de fala, score de clickbait e emojis.
    """
    print("Iniciando Etapa 5: Processando Engenharia de Formato e Copywriting...")

    try:
        # 1. Timeline do Roteiro (estrutura_blocos)
        condicoes = [
            df['duracao_segundos'] <= 30,
            (df['duracao_segundos'] > 30) & (df['duracao_segundos'] <= 60),
            df['duracao_segundos'] > 60
        ]
        escolhas = [
            'bloco_unico_impacto',
            'hook_desenvolvimento_punchline',
            'hook_narrativa_densa_cta'
        ]
        df['estrutura_blocos'] = np.select(condicoes, escolhas, default='desconhecido')

        # 2. Pacing (ritmo_palavras_seg)
        # Conta palavras separando por espaço. Se estiver nulo, conta como 0.
        contagem_palavras = df['texto_falado'].fillna('').apply(lambda x: len(str(x).split()))
        df['ritmo_palavras_seg'] = np.where(
            df['duracao_segundos'] > 0,
            contagem_palavras / df['duracao_segundos'],
            0
        )

        # 3. Sonoplastia (pistas_audio)
        def extrair_audio(texto):
            texto_str = str(texto)
            if texto_str == 'nan' or not texto_str.strip():
                return ""
            # Busca tudo que está entre [] ou ()
            matches = re.findall(r'\[.*?\]|\(.*?\)', texto_str)
            return " ".join(matches) if matches else ""

        # Aplica no texto falado e no título, juntando os dois
        df['pistas_audio'] = (df['texto_falado'].apply(extrair_audio) + " " + df['titulo'].apply(extrair_audio)).str.strip()

        # 4. Copywriting (clickbait_score)
        def calc_caixa_alta(texto):
            texto_str = str(texto)
            if texto_str == 'nan' or not texto_str.strip(): 
                return 0.0
            # Conta letras maiúsculas e o total de letras (ignorando números/símbolos)
            caps = sum(1 for c in texto_str if c.isupper())
            letras = sum(1 for c in texto_str if c.isalpha())
            return (caps / letras * 100) if letras > 0 else 0.0

        df['clickbait_score'] = df['titulo'].apply(calc_caixa_alta)

        # 5. Emoção (vibe_emojis)
        def extrair_emojis(texto):
            texto_str = str(texto)
            if texto_str == 'nan' or not texto_str.strip():
                return ""
            # Pega todos os emojis da string usando a biblioteca oficial
            emojis_encontrados = [c['emoji'] for c in emoji.emoji_list(texto_str)]
            return "".join(emojis_encontrados)

        # Junta emojis do título e da descrição
        df['vibe_emojis'] = (df['titulo'].apply(extrair_emojis) + df['descricao'].apply(extrair_emojis))

        # 6. SEO (tags_limpas)
        def limpar_tags(tag_val):
            # Blindagem que combinamos para evitar o crash com as listas da API
            if isinstance(tag_val, (list, np.ndarray)):
                return ", ".join(str(t) for t in tag_val)
            
            tag_str = str(tag_val)
            if tag_str == 'nan' or tag_str.strip() == "":
                return ""
            
            try:
                # Tenta converter a string "['tag1', 'tag2']" de volta para uma lista real do Python
                if tag_str.startswith('['):
                    tags_list = ast.literal_eval(tag_str)
                    return ", ".join(tags_list)
            except(ValueError, SyntaxError):
                pass
            # Fallback caso a avaliação falhe: remove colchetes e aspas com Regex
            return re.sub(r"[\[\]\']", "", tag_str)

        df['tags_limpas'] = df['tags'].apply(limpar_tags)

        print(" SUCESSO ABSOLUTO!")
        print(" -> Novas colunas adicionadas:")
        print("    [Estrutura]: estrutura_blocos, ritmo_palavras_seg")
        print("    [Direção]: pistas_audio, clickbait_score, vibe_emojis")
        print("    [SEO]: tags_limpas")
        print("\nAs instruções de Direção e Roteiro estão prontas para o LLM!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 5: {e}")

    return df

def criar_assinaturas_semanticas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera Assinaturas Semânticas (Super Match) combinando variáveis.
    Prepara os dados para indexação vetorial e leitura nativa pelo LLM.
    """
    print("Iniciando Etapa 6: Gerando Assinaturas Semânticas (Super Match)...")

    try:
        # Funções auxiliares para traduzir números em "conceitos textuais" para a IA
        def categorizar_ritmo(ritmo):
            if pd.isna(ritmo):
                return "desconhecido"
            if ritmo < 1.0:
                return "baixo"
            elif ritmo < 2.5:
                return "médio"
            else:
                return "alto"

        def formatar_views(views):
            if pd.isna(views):
                return "0"
            if views >= 1000:
                return f"{int(views/1000)}k"
            return str(int(views))

        def formatar_booleano(valor):
            return "Sim" if valor else "Não"

        # Função principal que constrói a frase para cada linha do dataset
        def criar_assinatura(row):
            nicho = str(row.get('nicho', 'N/A'))

            # Limpa formatação de lista dos tópicos, se houver
            topicos = str(row.get('topicos_wikipedia', '')).replace('[', '').replace(']', '').replace("'", "")

            formato = str(row.get('duracao_segundos', 0))
            timeline = str(row.get('estrutura_blocos', 'N/A'))
            ritmo = categorizar_ritmo(row.get('ritmo_palavras_seg', 0))
            tracao = formatar_views(row.get('velocidade_views', 0))
            audio = formatar_booleano(row.get('conteudo_licenciado', False))
            tags = str(row.get('tags_limpas', ''))

            # Montando o mega prompt (A Assinatura)
            assinatura = (
                f"Nicho: [{nicho}]. "
                f"Tópicos: [{topicos}]. "
                f"Formato: [{formato}] segundos. "
                f"Timeline: [{timeline}]. "
                f"Ritmo: [{ritmo}]. "
                f"Tração: [{tracao}] views/dia. "
                f"Áudio trend: [{audio}]. "
                f"Tags: [{tags}]."
            )
            return assinatura

        # Aplica a função linha a linha
        df['assinatura_vetorial'] = df.apply(criar_assinatura, axis=1)

        print(" SUCESSO ABSOLUTO!")
        print(" -> Coluna criada: assinatura_vetorial")
        
        # Pequena trava de segurança para o print não quebrar se o df vier vazio
        if not df.empty:
            print("\n💡 Exemplo de como a IA vai enxergar o primeiro vídeo:")
            print(f"   {df['assinatura_vetorial'].iloc[0]}")
            
        print("\nO Vector Store agora tem a 'Isca Perfeita' para a Partida Fria!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 6: {e}")

    return df

def purificar_texto_falado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purificação Semântica do Texto.
    Remove conteúdo entre colchetes [] ou parênteses () (como tags de áudio)
    e cria a coluna 'texto_falado_limpo' ideal para geração de Embeddings.
    """
    print("Iniciando Etapa 7: Purificação Semântica do Texto...")

    try:
        # Função com Regex GERAL: Remove qualquer conteúdo entre [] ou ()
        def purificar_texto(texto):
            if pd.isna(texto):
                return ""
            texto_str = str(texto)

            # Substitui [...] ou (...) por um espaço vazio
            texto_sem_tags = re.sub(r'\[.*?\]|\(.*?\)', ' ', texto_str)

            # Limpa os espaços duplos que podem ter sobrado e remove espaços nas pontas
            texto_limpo = re.sub(r'\s+', ' ', texto_sem_tags).strip()

            return texto_limpo

        # Cria a nova coluna limpa (preservando o pipeline)
        if 'texto_falado' in df.columns:
            df['texto_falado_limpo'] = df['texto_falado'].apply(purificar_texto)

        print(" SUCESSO ABSOLUTO!")
        print(" -> Nova coluna criada: texto_falado_limpo")
        print(" -> Todas as tags de áudio e anotações do YouTube foram varridas.")
        print("\nO texto agora é puramente narrativo e está pronto para o Embedding!")

    except Exception as e:
        print(f" Ops, deu um erro ao tentar processar a Etapa 7: {e}")

    return df

def unificar_idioma_ingles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unificação de Idioma (Arquitetura Híbrida sem perda de dados).
    Traduz as variáveis principais de texto para o inglês usando chunking,
    evitando que textos longos quebrem o limite da API (5000 chars).
    """
    # Inicializa o tqdm para o pandas
    tqdm.pandas()

    print("Iniciando Etapa 8: Unificação de Idioma (Arquitetura Híbrida sem perda de dados)...")

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

        # Aplicando a tradução segura (com checagem de segurança para não quebrar)
        print("1/4: Traduzindo a Assinatura Vetorial...")
        if 'assinatura_vetorial' in df.columns:
            df['assinatura_vetorial_en'] = df['assinatura_vetorial'].progress_apply(traduzir_texto_sem_perda)

        print("\n2/4: Traduzindo o Texto Falado Limpo...")
        if 'texto_falado_limpo' in df.columns:
            df['texto_falado_limpo_en'] = df['texto_falado_limpo'].progress_apply(traduzir_texto_sem_perda)

        print("\n3/4: Traduzindo os Títulos...")
        if 'titulo' in df.columns:
            df['titulo_en'] = df['titulo'].progress_apply(traduzir_texto_sem_perda)

        print("\n4/4: Traduzindo as Descrições...")
        if 'descricao' in df.columns:
            df['descricao_en'] = df['descricao'].progress_apply(traduzir_texto_sem_perda)

        print("\n SUCESSO ABSOLUTO! Sem nenhum dado perdido.")

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
    Recebe o DataFrame (a linha da Salinha VIP) bruto e devolve processado.
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
    
    # Etapa 6: Assinaturas Semânticas (Vector Store)
    df = criar_assinaturas_semanticas(df)
    
    # Etapa 7: Purificação do Texto Falado
    df = purificar_texto_falado(df)
    
    # Etapa 8: Unificação de Idioma (Tradução segura)
    df = unificar_idioma_ingles(df)
    
    # Etapa 9: Visão Computacional (Gemini 2.5 Flash)
    df = extrair_features_thumbnail_gemini(df)

    print("\n" + "="*55)
    print("🏁 USINA DE FEATURES FINALIZADA COM SUCESSO!")
    print("="*55 + "\n")

    return df