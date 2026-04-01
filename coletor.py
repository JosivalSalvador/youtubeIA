import os
import pandas as pd
import isodate
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Importa a função de extração do seu outro arquivo .py
# (Substitua 'nome_do_arquivo_extrator' pelo nome real do seu arquivo)
from extrator_legendas import extrair_texto_falado

# --- CONFIGURAÇÃO INICIAL DO MÓDULO ---
# Carrega a chave do .env de forma isolada e segura
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

if not API_KEY:
    raise ValueError("Erro Crítico: YOUTUBE_API_KEY não encontrada no arquivo .env")

# Constrói o cliente da API Oficial do Google
youtube = build('youtube', 'v3', developerKey=API_KEY)

# --- FUNÇÃO PRINCIPAL DE COLETA (MANTIDA 100% FIEL AO SEU MAPEAMENTO) ---
def buscar_dados_completos_shorts(query, max_results=50):
    """
    Realiza a coleta de metadados de vídeos do YouTube com base em uma string de busca.
    """
    print(f"\nIniciando busca na API Oficial para a query: '{query}'")
    
    try:
        video_ids = []
        next_page_token = None

        # Etapa 1: Coleta de IDs de vídeos utilizando paginação (pageToken)
        while len(video_ids) < max_results:
            limite_atual = min(50, max_results - len(video_ids))

            search_kwargs = {
                "q": query,
                "part": "id",
                "type": "video",
                "videoDuration": "short", # Garante que são Shorts
                "order": "viewCount",
                "maxResults": limite_atual,
            }
            if next_page_token:
                search_kwargs["pageToken"] = next_page_token

            request_search = youtube.search().list(**search_kwargs)
            response_search = request_search.execute()

            novos_ids = [item['id']['videoId'] for item in response_search.get('items', []) if item['id'].get('videoId')]
            video_ids.extend(novos_ids)

            next_page_token = response_search.get('nextPageToken', None)

            if not next_page_token or not novos_ids:
                break

        video_ids = video_ids[:max_results]

        if not video_ids:
            print("Resultado da consulta: Nenhum vídeo encontrado.")
            return pd.DataFrame()

        videos_data = []

        # Divisão dos IDs em lotes de até 50 elementos (limite do endpoint videos().list)
        lotes_ids = [video_ids[i:i + 50] for i in range(0, len(video_ids), 50)]

        # Etapa 2: Requisição em lote para coleta exaustiva de atributos
        for lote in lotes_ids:
            request_videos = youtube.videos().list(
                part="snippet,contentDetails,statistics,status,topicDetails,recordingDetails,liveStreamingDetails",
                id=",".join(lote)
            )
            response_videos = request_videos.execute()

            # Etapa 3: Mapeamento e extração com controle de fluxo (Delay)
            for item in tqdm(response_videos.get("items", []), desc="Extraindo dados e legendas"):
                video_id = item.get("id")
                snippet = item.get("snippet", {})
                content = item.get("contentDetails", {})
                stats = item.get("statistics", {})
                status = item.get("status", {})
                topics = item.get("topicDetails", {})
                recording = item.get("recordingDetails", {})
                livestream = item.get("liveStreamingDetails", {})

                duracao_iso = content.get("duration", "PT0S")
                try:
                    duracao_segundos = int(isodate.parse_duration(duracao_iso).total_seconds())
                except Exception:
                    duracao_segundos = 0

                # Chamada da função de integração de texto com Jitter Anti-Bot
                try:
                    texto_falado = extrair_texto_falado(video_id)
                except Exception as e:
                    texto_falado = f"[ERRO_SISTEMICO_EXTRAÇÃO]: {str(e)}"

                # O SEU MAPEAMENTO COMPLETO DE DADOS INTACTO
                video_info = {
                    "video_id": video_id,
                    "titulo": snippet.get("title", ""),
                    "descricao": snippet.get("description", ""),
                    "tags": snippet.get("tags", []),
                    "texto_falado": texto_falado,
                    "canal_id": snippet.get("channelId", ""),
                    "canal_nome": snippet.get("channelTitle", ""),
                    "visualizacoes": int(stats.get("viewCount", 0)),
                    "curtidas": int(stats.get("likeCount", 0)),
                    "comentarios": int(stats.get("commentCount", 0)),
                    "favoritos": int(stats.get("favoriteCount", 0)),
                    "data_publicacao": snippet.get("publishedAt", ""),
                    "data_gravacao": recording.get("recordingDate", ""),
                    "local_gravacao_desc": recording.get("locationDescription", ""),
                    "latitude": recording.get("location", {}).get("latitude", ""),
                    "longitude": recording.get("location", {}).get("longitude", ""),
                    "duracao_iso": duracao_iso,
                    "duracao_segundos": duracao_segundos,
                    "dimensao": content.get("dimension", ""),
                    "definicao": content.get("definition", ""),
                    "tem_legenda_nativa": content.get("caption", ""),
                    "projecao": content.get("projection", ""),
                    "conteudo_licenciado": content.get("licensedContent", False),
                    "privacidade": status.get("privacyStatus", ""),
                    "licenca": status.get("license", ""),
                    "permite_embed": status.get("embeddable", False),
                    "estatisticas_publicas": status.get("publicStatsViewable", False),
                    "feito_para_criancas": status.get("madeForKids", False),
                    "restricao_idade": content.get("contentRating", {}).get("ytRating", ""),
                    "restricao_regiao_bloqueada": content.get("regionRestriction", {}).get("blocked", []),
                    "restricao_regiao_permitida": content.get("regionRestriction", {}).get("allowed", []),
                    "categoria_id": snippet.get("categoryId", ""),
                    "idioma_audio_default": snippet.get("defaultAudioLanguage", ""),
                    "idioma_texto_default": snippet.get("defaultLanguage", ""),
                    "topicos_wikipedia": topics.get("topicCategories", []),
                    "topicos_ids": topics.get("topicIds", []),
                    "thumb_default": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                    "thumb_maxres": snippet.get("thumbnails", {}).get("maxres", {}).get("url", ""),
                    "live_inicio_real": livestream.get("actualStartTime", ""),
                    "live_fim_real": livestream.get("actualEndTime", ""),
                    "live_inicio_agendado": livestream.get("scheduledStartTime", ""),
                    "live_fim_agendado": livestream.get("scheduledEndTime", ""),
                    "espectadores_simultaneos": livestream.get("concurrentViewers", "")
                }
                videos_data.append(video_info)

        return pd.DataFrame(videos_data)

    except Exception as e:
        print(f"Erro sistêmico na requisição da API: {e}")
        return pd.DataFrame()