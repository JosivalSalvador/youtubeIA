import pandas as pd
import re
import emoji
# Importa a sua função do outro arquivo (troque 'nome_do_seu_arquivo' pelo nome real)
from coletor import buscar_dados_completos_shorts

print("Iniciando coleta massiva (com sinônimos) e diagnóstico geral para os 3 nichos...\n")

# A GRANDE MUDANÇA: Mapeando os cenários com múltiplas palavras-chave (Sinônimos)
# Se o YouTube esgotar a primeira palavra na página 1, ele puxa a próxima até bater a meta!
nichos_teste = {
    "Nicho 1 (Esquetes)": ["comedy skit shorts", "funny sketch shorts", "comedy shorts"],
    "Nicho 2 (Satisfying/Zero)": ["oddly satisfying shorts", "satisfying video shorts", "ASMR satisfying shorts", "kinetic sand shorts"],
    "Nicho 3 (Histórias/Denso)": ["reddit stories shorts", "storytime shorts", "askreddit shorts"]
}

dataframes_nichos = {}
META_POR_NICHO = 100 # Define a quantidade exata que queremos por nicho

for nome_nicho, lista_queries in nichos_teste.items():
    print(f"--- Coletando dados para: {nome_nicho} ---")

    df_nicho_acumulado = pd.DataFrame()
    videos_coletados_nicho = 0

    # Loop inteligente: Navega pelas palavras-chave até bater os 100 vídeos
    for query in lista_queries:
        if videos_coletados_nicho >= META_POR_NICHO:
            break # Já bateu a meta, não precisa gastar cota da API

        faltam = META_POR_NICHO - videos_coletados_nicho
        print(f"  > Buscando termo: '{query}' (Faltam {faltam} vídeos)")

        df_temp = buscar_dados_completos_shorts(query=query, max_results=faltam)

        if not df_temp.empty:
            df_nicho_acumulado = pd.concat([df_nicho_acumulado, df_temp], ignore_index=True)
            # Removemos possíveis duplicatas caso duas palavras-chave tragam o mesmo vídeo
            df_nicho_acumulado = df_nicho_acumulado.drop_duplicates(subset=['video_id'], keep='first')
            videos_coletados_nicho = len(df_nicho_acumulado)

    if not df_nicho_acumulado.empty:
        df_nicho_acumulado['nicho'] = nome_nicho # Adiciona a coluna marcando a origem
        dataframes_nichos[nome_nicho] = df_nicho_acumulado
        print(f" ✓ SUCESSO: {len(df_nicho_acumulado)} vídeos consolidados para {nome_nicho}.\n")
    else:
        print(f" Falha total na coleta do nicho {nome_nicho}.\n")

# Consolidando o resultado final
if dataframes_nichos:
    df_teste_final = pd.concat(dataframes_nichos.values(), ignore_index=True)

    print("\n DIAGNÓSTICO DE SUJEIRA TEXTUAL (O que precisamos limpar no próximo passo):")
    print("-" * 80)

    # Analisando apenas uma amostra (os 5 primeiros) para não poluir o terminal inteiro com 300 linhas
    for index, row in df_teste_final.head(5).iterrows():
        texto = str(row['texto_falado'])

        if texto.startswith("[ERRO"):
            status = " SEM TEXTO"
            detalhes = "Apenas log de erro do yt-dlp."
        else:
            status = " CAPTURADO"
            qtd_emojis = emoji.emoji_count(texto)
            tags_sistema = re.findall(r'\[.*?\]', texto)
            tem_links = bool(re.search(r'http\S+|www\.\S+', texto))
            detalhes = f"Tamanho: {len(texto)} chars | Emojis: {qtd_emojis} | Tags YT: {len(tags_sistema)} {tags_sistema[:3]} | Links: {tem_links}"

        print(f"Vídeo {index+1} [{row['nicho']}]: {status}\n   -> {detalhes}\n")

    print("\n VISUALIZAÇÃO DE TODOS OS METADADOS COLETADOS (Amostra):")

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    df_com_texto = df_teste_final[~df_teste_final['texto_falado'].str.startswith("[ERRO", na=False)]
    
    # Única mudança aqui: display() não existe fora do Colab, substituído por print()
    print(df_com_texto.head(3))

    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.max_columns')

    print("\n MAPA DE VARIÁVEIS: TODAS AS COLUNAS DISPONÍVEIS PARA A ETAPA 5:")
    print(df_teste_final.columns.tolist())

    # Salva os dados brutos fisicamente em um arquivo CSV
    df_teste_final.to_csv("dados_brutos_youtube.csv", index=False, encoding='utf-8')
    print("\n[+] Coleta finalizada! Dados salvos com sucesso no arquivo 'dados_brutos_youtube.csv'.")