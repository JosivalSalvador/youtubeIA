import re
from youtube_transcript_api import YouTubeTranscriptApi

def extrair_texto_falado(video_id):
    """
    Substitui o yt-dlp pela youtube-transcript-api.
    Lógica de fallback de idiomas e limpeza de texto mantida.
    Prioridade ajustada para capturar a transcrição REAL (Automática > Manual).
    O controle de tempo (pacing) foi movido para o orquestrador para permitir processamento assíncrono.
    """
    try:
        # 1. NOVA SINTAXE DE LISTAGEM:
        # Instanciar a classe e chamar .list()
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        en_manual = []
        en_auto = []
        any_manual = []
        any_auto = []

        # Analisa o contexto do que está disponível e separa por tipo
        for transcript in transcript_list:
            if transcript.language_code.startswith('en'):
                if not transcript.is_generated:
                    en_manual.append(transcript)
                else:
                    en_auto.append(transcript)
            
            if not transcript.is_generated:
                any_manual.append(transcript)
            else:
                any_auto.append(transcript)

        # NOVA ORDEM DE PRIORIDADE: Foco no áudio real (Automática primeiro)
        melhor_legenda = None
        if en_auto:
            melhor_legenda = en_auto[0]
        elif en_manual:
            melhor_legenda = en_manual[0]
        elif any_auto:
            melhor_legenda = any_auto[0]
        elif any_manual:
            melhor_legenda = any_manual[0]

        if not melhor_legenda:
            return "[ERRO_TRANSCRIÇÃO]: Nenhuma legenda suportada encontrada neste vídeo."

        # 2. NOVA SINTAXE DE EXTRAÇÃO:
        # fetch() agora retorna objetos FetchedTranscriptSnippet
        dados_legenda = melhor_legenda.fetch()

        # 3. ACESSO VIA ATRIBUTO (.text) EM VEZ DE DICIONÁRIO (['text'])
        texto_completo = [seg.text for seg in dados_legenda]
        texto_final = " ".join(texto_completo)

        # Limpeza final idêntica ao seu código original
        texto_limpo = re.sub(r'\s+', ' ', texto_final).strip()
        texto_limpo = texto_limpo.replace('\n', ' ')
        
        return texto_limpo

    except Exception as e:
        erro_str = str(e)
        
        # --- NOVO: IDENTIFICADOR CRÍTICO DE BLOQUEIO DE IP ---
        # Se for o Erro 429, ele manda um sinal claro para o coletor.py parar de chamar essa função
        if "429" in erro_str or "Too Many Requests" in erro_str or "IpBlocked" in erro_str:
            return "[ERRO_CRITICO_IP_BLOQUEADO]"

        # Mantém exatamente o seu padrão de tratamento de string de erro
        erro_msg = str(e).strip().split('\n')[0][:80]
        tipo_erro = type(e).__name__
        return f"[ERRO_{tipo_erro}]: {erro_msg}"