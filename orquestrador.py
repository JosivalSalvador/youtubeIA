import os
import time
import random
import pandas as pd

# Importações dos nossos módulos blindados
from conexoes_externas import (
    baixar_csv_mestre, 
    atualizar_csv_mestre, 
    limpar_rastro_local, 
    obter_nichos_ativos,
    NOME_ARQUIVO_MESTRE,
    NOME_ARQUIVO_FILA
)
from coletor_youtube import buscar_dados_completos_shorts
from extrator_legendas import extrair_texto_falado
from processador_features import maestro_features

def gerenciar_teto_200(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém estritamente os 200 vídeos mais recentes por nicho.
    Remove os mais antigos (com base na data de publicação) para evitar inchaço da base.
    """
    if df.empty:
        return df
    
    # Garante que a data está em formato datetime para ordenação
    df['data_publicacao'] = pd.to_datetime(df['data_publicacao'], utc=True)
    
    # Ordena: Nicho em ordem, Data decrescente (mais novo primeiro)
    df_ordenado = df.sort_values(by=['nicho', 'data_publicacao'], ascending=[True, False])
    
    # Agrupa por nicho e corta no limite de 200
    df_limitado = df_ordenado.groupby('nicho').head(200).reset_index(drop=True)
    
    return df_limitado

def main():
    print("=== INICIANDO PIPELINE DE ENGENHARIA REVERSA YOUTUBE (MODO FÍSICO) ===\n")

    # ====================================================================
    # GESTÃO DE ESTADO: VERIFICAÇÃO DE CRASH RECOVERY
    # ====================================================================
    modo_resgate = False
    if os.path.exists(NOME_ARQUIVO_FILA) and os.path.exists(NOME_ARQUIVO_MESTRE):
        df_fila_teste = pd.read_csv(NOME_ARQUIVO_FILA)
        if not df_fila_teste.empty:
            print("🚨 ATENÇÃO: Fila pendente detectada no disco rígido!")
            print(f"🔄 MODO RESGATE ATIVADO: Retomando processamento de {len(df_fila_teste)} vídeos da última sessão interrompida.")
            modo_resgate = True
            
            # Carrega o mestre LOCAL (NÃO baixa do Supabase para não sobrescrever os que já foram processados na queda)
            df_mestre = pd.read_csv(NOME_ARQUIVO_MESTRE)

    if not modo_resgate:
        # 1. INÍCIO LIMPO: DOWNLOAD DA BASE EXISTENTE
        df_mestre = baixar_csv_mestre()
        
        # Como o CSV base sempre vai existir, pulamos validações redundantes
        if 'texto_falado' not in df_mestre.columns:
            df_mestre['texto_falado'] = ""
            
        # Cria a memória ultra-rápida direto dos dados que já existem (Escudo Anti-Duplicata)
        ids_conhecidos = set(df_mestre['video_id'].dropna().unique())

        print(f"[Orquestrador] Mestre baixado. Inventário: {len(ids_conhecidos)} vídeos já conhecidos.")

        # ====================================================================
        # FASE 1: COLETA DE METADADOS INÉDITOS (Batedor)
        # ====================================================================
        nichos_ativos = obter_nichos_ativos()
        META_POR_NICHO = 25 
        
        print("\n--- FASE 1: VARREDURA DE NOVOS VÍDEOS ---")
        novos_dados_lista = []

        for query_nicho in nichos_ativos:
            print(f"\nColetando alvo: {query_nicho}")
            
            # O Coletor já lida com a filtragem de duplicatas usando o ids_conhecidos
            df_temp = buscar_dados_completos_shorts(query=query_nicho, ids_conhecidos=ids_conhecidos, max_results=META_POR_NICHO)

            if not df_temp.empty:
                df_temp['nicho'] = query_nicho 
                novos_dados_lista.append(df_temp)

        # Salva a Fila Fisicamente no HD
        if novos_dados_lista:
            df_novos = pd.concat(novos_dados_lista, ignore_index=True)
            df_novos['texto_falado'] = "" 
            
            # O COFRE: Salva no disco rígido antes de processar qualquer coisa
            df_novos.to_csv(NOME_ARQUIVO_FILA, index=False)
            print(f"\n[Orquestrador] ✓ Fase 1 Concluída. {len(df_novos)} novos vídeos trancados no cofre '{NOME_ARQUIVO_FILA}'.")
        else:
            print("\n[Orquestrador] Nenhum vídeo novo adicionado nesta rodada.")
            # Se não tem fila e não é resgate, o script pode pular pro final
            if os.path.exists(NOME_ARQUIVO_FILA):
                os.remove(NOME_ARQUIVO_FILA)

    # ====================================================================
    # FASE 2: A SALINHA VIP (Fila Contínua no Disco)
    # ====================================================================
    print("\n--- FASE 2: USINA DE PROCESSAMENTO (LINHA POR LINHA) ---")
    
    while True:
        # 1. Tenta abrir a fila física
        if not os.path.exists(NOME_ARQUIVO_FILA):
            print("Fila pendente não encontrada. Seguindo para o encerramento.")
            break
            
        df_fila = pd.read_csv(NOME_ARQUIVO_FILA)
        
        if df_fila.empty:
            print("Fila pendente esvaziada por completo! Todos processados.")
            break

        print(f"\n=> Status da Fila: {len(df_fila)} vídeos aguardando.")
        
        # 2. O "Pop" - Arranca a primeira linha (Salinha VIP)
        linha_vip = df_fila.iloc[[0]].copy()
        video_id = linha_vip['video_id'].values[0]
        
        print(f"🎬 [Processando VIP: {video_id}]")

        tempo_total_espera = random.uniform(120, 300)
        print(f"   > Meta de pacing anti-bot: {tempo_total_espera/60:.1f} minutos.")
        inicio_relogio = time.time()

        # 3. Extrai a legenda do YouTube
        texto = extrair_texto_falado(video_id)
        linha_vip['texto_falado'] = texto

        if texto == "[ERRO_CRITICO_IP_BLOQUEADO]":
            print("🚨 [ALERTA CRÍTICO] O YouTube bloqueou o IP! Abortando a Usina agora.")
            print("🛡️ O seu progresso até aqui está salvo no HD. Rode o script amanhã para continuar do resgate.")
            break 

        # 4. CHAMA O MAESTRO (As 9 etapas da Usina)
        linha_pronta = maestro_features(linha_vip)

        # ======================================================
        # 🔍 INSPEÇÃO VISUAL DA LINHA ANTES DE SALVAR
        # ======================================================
        print("\n" + "🔎 "*15)
        print("RAIO-X DO VÍDEO PROCESSADO:")
        print(linha_pronta.T) # O .T vira as colunas em linhas para facilitar a leitura
        print("🔎 "*15 + "\n")
        # ======================================================

        # 5. O CHECKPOINT: Salva no master local fisicamente
        df_mestre = pd.concat([df_mestre, linha_pronta], ignore_index=True)
        df_mestre.to_csv(NOME_ARQUIVO_MESTRE, index=False)
        print("   💾 Checkpoint salvo: Vídeo sólido adicionado ao master local.")

        # 6. Atualiza a fila removendo o vídeo processado e salvando no HD
        df_fila = df_fila.iloc[1:]
        df_fila.to_csv(NOME_ARQUIVO_FILA, index=False)

        # 7. Matemática do Tempo
        tempo_gasto_ate_agora = time.time() - inicio_relogio
        tempo_restante = tempo_total_espera - tempo_gasto_ate_agora

        if tempo_restante > 0:
            print(f"   💤 Processamento rápido. Descansando o restante ({tempo_restante/60:.1f} min)...")
            time.sleep(tempo_restante)
        else:
            print("   ⚡ O processamento da Usina consumiu todo o tempo! Próximo da fila.")


    # ====================================================================
    # FASE 3: GESTÃO DE ESTOQUE, GUILHOTINA E UPLOAD
    # ====================================================================
    # Só fazemos o upload e a limpeza se a fila esvaziou de verdade (não houve bloqueio de IP)
    if os.path.exists(NOME_ARQUIVO_FILA):
        df_verificacao = pd.read_csv(NOME_ARQUIVO_FILA)
        if not df_verificacao.empty:
            print("\n⚠️ AVISO: A operação foi interrompida antes do fim da fila. O upload para o Supabase foi adiado.")
            print("   Seus dados estão protegidos no HD. O script continuará no Modo Resgate na próxima vez.")
            return # Sai do programa sem subir e sem limpar o disco

    print("\n--- FASE 3: GESTÃO DE ESTOQUE E UPLOAD ---")
    
    # Remove qualquer possível duplicata residual
    df_mestre = df_mestre.drop_duplicates(subset=['video_id'], keep='last')
    
    # Aplica a guilhotina matemática (Corta as cabeças dos mais antigos se passar de 200)
    df_mestre = gerenciar_teto_200(df_mestre)

    # ÚNICO UPLOAD MATADOR DE TODA A OPERAÇÃO
    print(f"   🚀 Subindo base consolidada e limpa para o Supabase ({len(df_mestre)} linhas totais)...")
    atualizar_csv_mestre(df_mestre)

    print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")
    
    # Diagnóstico Final
    df_com_texto = df_mestre[
        (df_mestre['texto_falado'] != "") & 
        (~df_mestre['texto_falado'].astype(str).str.startswith("[ERRO", na=False))
    ]
    print(f"📈 Diagnóstico: O banco possui agora {len(df_mestre)} vídeos virais blindados, sendo {len(df_com_texto)} com transcrições purificadas.")

    # A Vassoura (Deleta o master local e a fila vazia)
    limpar_rastro_local()

if __name__ == "__main__":
    main()