import os
import pandas as pd
import isodate
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone

# --- CONFIGURAÇÃO INICIAL DO MÓDULO ---
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

if not API_KEY:
    raise ValueError("Erro Crítico: YOUTUBE_API_KEY não encontrada no arquivo .env")

youtube = build('youtube', 'v3', developerKey=API_KEY)

# --- FUNÇÃO PRINCIPAL DE COLETA (AJUSTADA PARA O PIPELINE) ---
def buscar_dados_completos_shorts(query, ids_conhecidos, max_results=25):
    """
    Realiza a coleta de metadados de vídeos do YouTube inéditos.
    O texto falado não é extraído aqui para permitir o pacing assíncrono no Orquestrador.
    """

    print(f"\nIniciando busca na API Oficial para a query: '{query}'")

    try:
        video_ids_ineditos = []

        # Etapa 1: Coleta de IDs com fallback progressivo de janela temporal
        # Tenta 3 meses → 4 meses → 5 meses até completar max_results
        for meses in [3, 4, 5]:
            if len(video_ids_ineditos) >= max_results:
                break

            dias = meses * 30
            data_limite = (datetime.now(timezone.utc) - timedelta(days=dias)).strftime('%Y-%m-%dT%H:%M:%SZ')
            print(f"📅 Tentando janela de {meses} meses (após {data_limite}) — inéditos até agora: {len(video_ids_ineditos)}/{max_results}")

            next_page_token = None

            while len(video_ids_ineditos) < max_results:
                # Puxa sempre 50 para ter margem de descarte contra os ids_conhecidos
                search_kwargs = {
                    "q": query,
                    "part": "id",
                    "type": "video",
                    "videoDuration": "short",  # Garante que são Shorts
                    "order": "viewCount",
                    "publishedAfter": data_limite,
                    "maxResults": 50,
                }
                if next_page_token:
                    search_kwargs["pageToken"] = next_page_token

                request_search = youtube.search().list(**search_kwargs)
                response_search = request_search.execute()

                novos_ids = [item['id']['videoId'] for item in response_search.get('items', []) if item['id'].get('videoId')]

                # Filtra os IDs verificando a base existente
                for vid in novos_ids:
                    if vid not in ids_conhecidos:
                        video_ids_ineditos.append(vid)
                        ids_conhecidos.add(vid)  # Evita duplicação na mesma run

                        if len(video_ids_ineditos) == max_results:
                            break  # Bateu a meta estabelecida

                next_page_token = response_search.get('nextPageToken', None)

                if not next_page_token:
                    break  # API sem mais páginas nessa janela — tenta a próxima

        if len(video_ids_ineditos) < max_results:
            print(f"⚠️ Esgotadas as 3 janelas. Coletados {len(video_ids_ineditos)}/{max_results} vídeos inéditos para '{query}'.")
        else:
            print(f"✓ Meta atingida: {len(video_ids_ineditos)} vídeos inéditos coletados para '{query}'.")

        if not video_ids_ineditos:
            print(f"Resultado da consulta: Nenhum vídeo inédito encontrado para '{query}' em nenhuma janela.")
            return pd.DataFrame()

        videos_data = []

        # Divisão dos IDs inéditos em lotes de até 50 elementos (limite do endpoint)
        lotes_ids = [video_ids_ineditos[i:i + 50] for i in range(0, len(video_ids_ineditos), 50)]

        # Etapa 2: Requisição em lote para coleta exaustiva de atributos
        for lote in lotes_ids:
            request_videos = youtube.videos().list(
                part="snippet,contentDetails,statistics,status,topicDetails,recordingDetails,liveStreamingDetails",
                id=",".join(lote)
            )
            response_videos = request_videos.execute()

            # Etapa 3: Mapeamento dos dados (Estrutura 100% mantida)
            for item in tqdm(response_videos.get("items", []), desc="Extraindo metadados"):
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

                video_info = {
                    "video_id": video_id,
                    "titulo": snippet.get("title", ""),
                    "descricao": snippet.get("description", ""),
                    "tags": snippet.get("tags", []),
                    "texto_falado": "",
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