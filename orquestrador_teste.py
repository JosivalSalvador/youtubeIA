import os
import sys
import time
import random
import datetime
import pandas as pd

# Importações dos nossos módulos blindados (mesmos do orquestrador original)
from conexoes_externas import (
    baixar_csv_mestre,
    obter_nichos_ativos,
    NOME_DO_BUCKET,
)
from coletor_youtube import buscar_dados_completos_shorts
from extrator_legendas import extrair_texto_falado
from processador_features import maestro_features

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES EXCLUSIVAS DO MODO TESTE
# ════════════════════════════════════════════════════════════════════════════
# Arquivos próprios, isolados do orquestrador de produção — evita qualquer
# risco de colisão/sobrescrita do NOME_ARQUIVO_MESTRE e NOME_ARQUIVO_FILA reais.
NOME_ARQUIVO_MESTRE_TESTE = "dataset_youtube_processado_modulo2_TESTE.csv"
NOME_ARQUIVO_FILA_TESTE = "fila_pendentes_teste.csv"

META_VIDEOS_TESTE = 3  # <- a única coisa que muda na meta de coleta

# ── Logging em arquivo (espelha tudo que vai pro terminal) ──────────────────
PASTA_LOGS_TESTE = "logs_teste"


class Tee:
    """
    Espelha tudo que é escrito em um stream (normalmente sys.stdout) para
    múltiplos destinos simultaneamente — aqui: o terminal real e um arquivo
    de log. Não substitui o terminal, só "ouve" o que passa por ele.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, dados):
        for stream in self.streams:
            stream.write(dados)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def gerenciar_teto_200(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém estritamente os 200 vídeos mais recentes por nicho.
    Remove os mais antigos (com base na data de publicação) para evitar inchaço da base.

    Idêntica à função do orquestrador original — mantida aqui só para preservar
    a estrutura/comportamento, mesmo que no teste o volume nunca chegue perto do teto.
    """
    if df.empty:
        return df

    # format='ISO8601' lida com strings em variações diferentes (ex: a coluna pode
    # ter linhas vindas direto da API do YouTube no formato "...T...Z" e linhas
    # vindas do CSV de produção relido, já no formato "... +00:00") sem quebrar
    # tentando inferir um único formato fixo para a coluna inteira.
    df['data_publicacao'] = pd.to_datetime(df['data_publicacao'], utc=True, format='ISO8601')

    df_ordenado = df.sort_values(by=['nicho', 'data_publicacao'], ascending=[True, False])

    df_limitado = df_ordenado.groupby('nicho').head(200).reset_index(drop=True)

    # Converte de volta para string no mesmo formato da API do YouTube
    df_limitado['data_publicacao'] = df_limitado['data_publicacao'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    return df_limitado


def comparar_estrutura(df_teste: pd.DataFrame, df_producao_amostra: pd.DataFrame):
    """
    Compara a ESTRUTURA dos dois DataFrames — não os valores.
    Verifica: nomes de colunas, ordem das colunas e dtype de cada coluna.
    Não compara conteúdo das células, só o "molde" do dado.
    """
    print("\n" + "=" * 70)
    print("🔬 COMPARAÇÃO DE ESTRUTURA — TESTE (3 vídeos) vs PRODUÇÃO (amostra de 3)")
    print("=" * 70)

    colunas_teste = list(df_teste.columns)
    colunas_producao = list(df_producao_amostra.columns)

    # ── 1. Mesma quantidade de colunas? ──────────────────────────────────────
    print(f"\nColunas no TESTE    : {len(colunas_teste)}")
    print(f"Colunas em PRODUÇÃO : {len(colunas_producao)}")

    if len(colunas_teste) != len(colunas_producao):
        print("⚠️  Quantidade de colunas DIFERENTE entre teste e produção!")
    else:
        print("✅ Mesma quantidade de colunas.")

    # ── 2. Mesmos nomes de coluna (ignorando ordem por enquanto) ────────────
    set_teste = set(colunas_teste)
    set_producao = set(colunas_producao)

    apenas_no_teste = set_teste - set_producao
    apenas_na_producao = set_producao - set_teste

    if apenas_no_teste:
        print(f"\n⚠️  Colunas presentes SÓ no teste ({len(apenas_no_teste)}):")
        for c in sorted(apenas_no_teste):
            print(f"     - {c}")

    if apenas_na_producao:
        print(f"\n⚠️  Colunas presentes SÓ na produção ({len(apenas_na_producao)}):")
        for c in sorted(apenas_na_producao):
            print(f"     - {c}")

    if not apenas_no_teste and not apenas_na_producao:
        print("✅ Mesmo conjunto de nomes de colunas.")

    # ── 3. Mesma ORDEM das colunas? ──────────────────────────────────────────
    if colunas_teste == colunas_producao:
        print("✅ Ordem das colunas IDÊNTICA.")
    else:
        print("⚠️  Ordem das colunas DIFERENTE. Detalhe posição a posição:")
        tamanho_max = max(len(colunas_teste), len(colunas_producao))
        for i in range(tamanho_max):
            col_t = colunas_teste[i] if i < len(colunas_teste) else "—"
            col_p = colunas_producao[i] if i < len(colunas_producao) else "—"
            marcador = "✅" if col_t == col_p else "❌"
            if marcador == "❌":
                print(f"   [{i:>3}] {marcador}  teste='{col_t}'  |  producao='{col_p}'")

    # ── 4. Comparação coluna a coluna: presença + dtype ──────────────────────
    print("\n── Detalhe por coluna (presença e tipo) ──")
    colunas_em_comum = [c for c in colunas_teste if c in colunas_producao]

    divergencias_tipo = []

    for col in colunas_em_comum:
        dtype_teste = str(df_teste[col].dtype)
        dtype_producao = str(df_producao_amostra[col].dtype)
        igual = dtype_teste == dtype_producao
        status = "✅" if igual else "❌"
        if not igual:
            divergencias_tipo.append((col, dtype_teste, dtype_producao))
        print(f"   {status} {col:<35} teste={dtype_teste:<12} producao={dtype_producao}")

    # ── 5. Resumo final ───────────────────────────────────────────────────────
    print("\n── RESUMO ──")
    estrutura_identica = (
        not apenas_no_teste
        and not apenas_na_producao
        and colunas_teste == colunas_producao
        and not divergencias_tipo
    )

    if estrutura_identica:
        print("✅ Estrutura BATE perfeitamente: mesmas colunas, mesma ordem, mesmos tipos.")
    else:
        print("⚠️  Estrutura NÃO bate 100%. Resumo dos problemas:")
        if apenas_no_teste or apenas_na_producao:
            print("   - Conjunto de colunas diferente (ver acima).")
        if colunas_teste != colunas_producao and not (apenas_no_teste or apenas_na_producao):
            print("   - Mesmas colunas, mas em ORDEM diferente.")
        if divergencias_tipo:
            print(f"   - {len(divergencias_tipo)} coluna(s) com dtype divergente:")
            for col, dt_t, dt_p in divergencias_tipo:
                print(f"       * {col}: teste={dt_t} vs producao={dt_p}")

    print("=" * 70 + "\n")


def main():
    # ── Liga o log em arquivo, sem desligar o terminal ───────────────────────
    os.makedirs(PASTA_LOGS_TESTE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    caminho_log = os.path.join(PASTA_LOGS_TESTE, f"log_teste_{timestamp}.txt")

    stdout_original = sys.stdout
    arquivo_log = open(caminho_log, "w", encoding="utf-8")
    sys.stdout = Tee(stdout_original, arquivo_log)

    print(f"📝 Log desta execução sendo salvo em: {caminho_log}\n")

    try:
        _executar_pipeline_teste()
    finally:
        # Restaura o stdout original mesmo se algo quebrar no meio do caminho,
        # senão o terminal fica "preso" escrevendo num arquivo já fechado.
        print(f"\n📝 Log completo salvo em: {caminho_log}")
        sys.stdout = stdout_original
        arquivo_log.close()


def _executar_pipeline_teste():
    print("=== INICIANDO PIPELINE DE TESTE — ENGENHARIA REVERSA YOUTUBE (MODO TESTE, SEM UPLOAD) ===\n")
    print(f"⚙️  Meta de coleta neste modo: {META_VIDEOS_TESTE} vídeos (apenas validação de estrutura).")
    print("🚫 Este script NUNCA sobe dados para o Supabase/produção.\n")

    # ====================================================================
    # GESTÃO DE ESTADO: VERIFICAÇÃO DE CRASH RECOVERY (arquivos de TESTE)
    # ====================================================================
    modo_resgate = False
    if os.path.exists(NOME_ARQUIVO_FILA_TESTE) and os.path.exists(NOME_ARQUIVO_MESTRE_TESTE):
        df_fila_teste_check = pd.read_csv(NOME_ARQUIVO_FILA_TESTE)
        if not df_fila_teste_check.empty:
            print("🚨 ATENÇÃO: Fila de TESTE pendente detectada no disco rígido!")
            print(f"🔄 MODO RESGATE ATIVADO: Retomando processamento de {len(df_fila_teste_check)} vídeos da última sessão de teste interrompida.")
            modo_resgate = True

            # Carrega o mestre de TESTE local (nunca baixa de produção em modo resgate,
            # mesma lógica do orquestrador original)
            df_mestre = pd.read_csv(NOME_ARQUIVO_MESTRE_TESTE)

    if not modo_resgate:
        # 1. INÍCIO LIMPO: baixa a base de PRODUÇÃO só como ponto de partida
        # (igual ao original — usado para herdar o "ids_conhecidos" e formato),
        # mas tudo que for coletado/processado aqui vai para os arquivos de TESTE.
        df_producao_base = baixar_csv_mestre()

        if 'texto_falado' not in df_producao_base.columns:
            df_producao_base['texto_falado'] = ""

        df_mestre = df_producao_base.copy()

        # Cria a memória ultra-rápida direto dos dados que já existem (Escudo Anti-Duplicata)
        ids_conhecidos = set(df_mestre['video_id'].dropna().unique())

        print(f"[Orquestrador-TESTE] Base de produção usada como referência. Inventário: {len(ids_conhecidos)} vídeos já conhecidos.")

        # ====================================================================
        # FASE 1: COLETA DE METADADOS INÉDITOS (Batedor) — só META_VIDEOS_TESTE
        # ====================================================================
        nichos_ativos = obter_nichos_ativos()

        print("\n--- FASE 1: VARREDURA DE NOVOS VÍDEOS (TESTE) ---")
        novos_dados_lista = []
        total_coletado = 0

        for query_nicho in nichos_ativos:
            if total_coletado >= META_VIDEOS_TESTE:
                break

            faltam = META_VIDEOS_TESTE - total_coletado
            print(f"\nColetando alvo: {query_nicho} (faltam {faltam} para a meta de teste)")

            df_temp = buscar_dados_completos_shorts(query=query_nicho, ids_conhecidos=ids_conhecidos, max_results=faltam)

            if not df_temp.empty:
                df_temp = df_temp.head(faltam)  # garante que nunca passa da meta de 3
                df_temp['nicho'] = query_nicho
                novos_dados_lista.append(df_temp)
                total_coletado += len(df_temp)

        # Salva a Fila Fisicamente no HD (arquivo de TESTE)
        if novos_dados_lista:
            df_novos = pd.concat(novos_dados_lista, ignore_index=True)
            df_novos['texto_falado'] = ""

            # O COFRE: Salva no disco rígido antes de processar qualquer coisa
            df_novos.to_csv(NOME_ARQUIVO_FILA_TESTE, index=False)
            print(f"\n[Orquestrador-TESTE] ✓ Fase 1 Concluída. {len(df_novos)} novos vídeos trancados no cofre de teste '{NOME_ARQUIVO_FILA_TESTE}'.")
        else:
            print("\n[Orquestrador-TESTE] Nenhum vídeo novo coletado nesta rodada.")
            if os.path.exists(NOME_ARQUIVO_FILA_TESTE):
                os.remove(NOME_ARQUIVO_FILA_TESTE)

        # ====================================================================
        # FASE 2: A SALINHA VIP (Fila Contínua no Disco) — arquivos de TESTE
        # ====================================================================
        print("\n--- FASE 2: USINA DE PROCESSAMENTO (LINHA POR LINHA) — TESTE ---")

        while True:
            if not os.path.exists(NOME_ARQUIVO_FILA_TESTE):
                print("Fila de teste pendente não encontrada. Seguindo para o encerramento.")
                break

            df_fila = pd.read_csv(NOME_ARQUIVO_FILA_TESTE)

            if df_fila.empty:
                print("Fila de teste esvaziada por completo! Todos processados.")
                break

            print(f"\n=> Status da Fila de Teste: {len(df_fila)} vídeos aguardando.")

            # 2. O "Pop" - Arranca a primeira linha (Salinha VIP)
            linha_vip = df_fila.iloc[[0]].copy()
            video_id = linha_vip['video_id'].values[0]

            print(f"🎬 [Processando VIP (TESTE): {video_id}]")

            # Pacing mantido idêntico ao orquestrador original (120-300s)
            tempo_total_espera = random.uniform(120, 300)
            print(f"   > Meta de pacing anti-bot: {tempo_total_espera/60:.1f} minutos.")
            inicio_relogio = time.time()

            # 3. Extrai a legenda do YouTube
            texto = extrair_texto_falado(video_id)
            linha_vip['texto_falado'] = texto

            if texto == "[ERRO_CRITICO_IP_BLOQUEADO]":
                print("🚨 [ALERTA CRÍTICO] O YouTube bloqueou o IP! Abortando a Usina de Teste agora.")
                print("🛡️ O seu progresso até aqui está salvo no HD (arquivos de teste). Rode o script novamente para continuar do resgate.")
                break

            # 4. CHAMA O MAESTRO (As mesmas etapas da Usina, sem alteração)
            linha_pronta = maestro_features(linha_vip)

            # ======================================================
            # 🔍 INSPEÇÃO VISUAL DA LINHA ANTES DE SALVAR
            # ======================================================
            print("\n" + "🔎 " * 15)
            print("RAIO-X DO VÍDEO PROCESSADO (TESTE):")
            print(linha_pronta.T)
            print("🔎 " * 15 + "\n")
            # ======================================================

            # 5. O CHECKPOINT: Salva no master de TESTE local fisicamente
            df_mestre = pd.concat([df_mestre, linha_pronta], ignore_index=True)
            df_mestre.to_csv(NOME_ARQUIVO_MESTRE_TESTE, index=False)
            print("   💾 Checkpoint salvo: Vídeo sólido adicionado ao master local de TESTE.")

            # 6. Atualiza a fila removendo o vídeo processado e salvando no HD
            df_fila = df_fila.iloc[1:]
            df_fila.to_csv(NOME_ARQUIVO_FILA_TESTE, index=False)

            # 7. Matemática do Tempo
            tempo_gasto_ate_agora = time.time() - inicio_relogio
            tempo_restante = tempo_total_espera - tempo_gasto_ate_agora

            if tempo_restante > 0:
                print(f"   💤 Processamento rápido. Descansando o restante ({tempo_restante/60:.1f} min)...")
                time.sleep(tempo_restante)
            else:
                print("   ⚡ O processamento da Usina consumiu todo o tempo! Próximo da fila.")

        # ====================================================================
        # FASE 3: GESTÃO DE ESTOQUE — SEM UPLOAD (diferença central do modo teste)
        # ====================================================================
        if os.path.exists(NOME_ARQUIVO_FILA_TESTE):
            df_verificacao = pd.read_csv(NOME_ARQUIVO_FILA_TESTE)
            if not df_verificacao.empty:
                print("\n⚠️ AVISO: A operação de teste foi interrompida antes do fim da fila.")
                print("   Seus dados estão protegidos no HD (arquivos de teste). O script continuará no Modo Resgate na próxima vez.")
                return

        print("\n--- FASE 3: GESTÃO DE ESTOQUE (TESTE) — SEM UPLOAD PARA PRODUÇÃO ---")

        # Remove qualquer possível duplicata residual
        df_mestre = df_mestre.drop_duplicates(subset=['video_id'], keep='last')

        # Aplica a guilhotina matemática (mantida igual, mesmo que aqui o volume seja pequeno)
        df_mestre = gerenciar_teto_200(df_mestre)

        print(f"   🧪 Base de TESTE consolidada localmente ({len(df_mestre)} linhas totais). NENHUM upload será feito.")

        print("\n=== PIPELINE DE TESTE FINALIZADO COM SUCESSO ===")

        # Diagnóstico Final
        df_com_texto = df_mestre[
            (df_mestre['texto_falado'] != "") &
            (~df_mestre['texto_falado'].astype(str).str.startswith("[ERRO", na=False))
        ]
        print(f"📈 Diagnóstico: A base de teste possui agora {len(df_mestre)} vídeos, sendo {len(df_com_texto)} com transcrições purificadas.")

        # ====================================================================
        # FASE 4: COMPARAÇÃO DE ESTRUTURA TESTE vs PRODUÇÃO
        # ====================================================================
        # Isola apenas as linhas que foram coletadas/processadas NESTA execução de teste
        # (os últimos vídeos adicionados ao master de teste), para comparar estrutura
        # "maçã com maçã" contra a amostra de produção.
        df_teste_apenas_novos = df_mestre.tail(min(META_VIDEOS_TESTE, len(df_mestre))).copy()

        print("\n--- FASE 4: VALIDAÇÃO DE ESTRUTURA CONTRA A PRODUÇÃO ---")
        print(f"🔽 Baixando CSV de produção ('{NOME_DO_BUCKET}') para comparação de estrutura...")
        df_producao_completo = baixar_csv_mestre()

        if df_producao_completo.empty:
            print("⚠️  Não foi possível baixar a base de produção — comparação de estrutura abortada.")
        elif df_teste_apenas_novos.empty:
            print("⚠️  Nenhum vídeo de teste foi processado — comparação de estrutura abortada.")
        else:
            qtd_amostra = min(3, len(df_producao_completo))
            df_producao_amostra = df_producao_completo.sample(n=qtd_amostra, random_state=None).reset_index(drop=True)

            print(f"🎲 Amostra aleatória de produção: {qtd_amostra} vídeo(s) sorteados para comparação.")

            comparar_estrutura(df_teste_apenas_novos.reset_index(drop=True), df_producao_amostra)

        # A Vassoura — limpa SOMENTE os arquivos de TESTE.
        # Nunca toca nos arquivos de produção (NOME_ARQUIVO_MESTRE / NOME_ARQUIVO_FILA reais).
        print("🧹 Limpando arquivos locais de TESTE...")
        if os.path.exists(NOME_ARQUIVO_MESTRE_TESTE):
            os.remove(NOME_ARQUIVO_MESTRE_TESTE)
            print(f"   Limpeza: '{NOME_ARQUIVO_MESTRE_TESTE}' apagado.")
        if os.path.exists(NOME_ARQUIVO_FILA_TESTE):
            os.remove(NOME_ARQUIVO_FILA_TESTE)
            print(f"   Limpeza: '{NOME_ARQUIVO_FILA_TESTE}' apagado.")


if __name__ == "__main__":
    main()