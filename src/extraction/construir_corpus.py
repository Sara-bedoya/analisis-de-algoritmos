"""Construye un CSV reproducible con metadatos y abstracts del corpus."""

import csv
import re
import sys
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ_PROYECTO))

from src.extraction.pdf_reader import extraer_abstract, extraer_texto_pdf
from src.preprocessing.limpieza_texto import normalizar_texto

CARPETA_ARTICULOS = RAIZ_PROYECTO / "data" / "articulos"
RUTA_SALIDA = RAIZ_PROYECTO / "data" / "corpus_articulos.csv"
TOTAL_ESPERADO = 20

# Metadatos cotejados con la portada de cada PDF. Se mantienen aquí porque
# los metadatos internos de varios PDF omiten coautores o tienen títulos erróneos.
METADATOS = {
    1: ("Artificial intelligence literacy in higher and adult education: A scoping literature review", "Matthias Carl Laupichler; Alexandra Aster; Jana Schirch; Tobias Raupach"),
    2: ("Systematic literature review on opportunities, challenges, and future research recommendations of artificial intelligence in education", "Thomas K.F. Chiu; Qi Xia; Xinyan Zhou; Ching Sing Chai; Miaoting Cheng"),
    3: ("Myths, mis- and preconceptions of artificial intelligence: A review of the literature", "Arne Bewersdorff; Xiaoming Zhai; Jessica Roberts; Claudia Nerdel"),
    4: ("Opportunities of artificial intelligence for supporting complex problem-solving: Findings from a scoping review", "Srecko Joksimovic; Dirk Ifenthaler; Rebecca Marrone; Maarten De Laat; George Siemens"),
    5: ("Application and theory gaps during the rise of Artificial Intelligence in Education", "Xieling Chen; Haoran Xie; Di Zou; Gwo-Jen Hwang"),
    6: ("A multi-perspective study on Artificial Intelligence in Education: grants, conferences, journals, software tools, institutions, and researchers", "Xieling Chen; Haoran Xie; Gwo-Jen Hwang"),
    7: ("Human-centred learning analytics and AI in education: A systematic literature review", "Riordan Alfredo; Vanessa Echeverria; Yueqiao Jin; Lixiang Yan; Zachari Swiecki; Dragan Gašević; Roberto Martinez-Maldonado"),
    8: ("A systematic review of artificial intelligence techniques for collaborative learning over the past two decades", "Seng Chee Tan; Alwyn Vwen Yen Lee; Min Lee"),
    9: ("AI technologies for education: Recent research & future directions", "Ke Zhang; Ayse Begum Aslan"),
    10: ("Educational applications of artificial intelligence in simulation-based learning: A systematic mapping review", "Chih-Pu Dai; Fengfeng Ke"),
    11: ("A scoping review on how generative artificial intelligence transforms assessment in higher education", "Qi Xia; Xiaojing Weng; Fan Ouyang; Tzung Jin Lin; Thomas K.F. Chiu"),
    12: ("A meta systematic review of artificial intelligence in higher education: a call for increased ethics, collaboration, and rigour", "Melissa Bond; Hassan Khosravi; Maarten De Laat; Nina Bergdahl; Violeta Negrea; Emily Oxley; Phuong Pham; Sin Wang Chong; George Siemens"),
    13: ("Artificial intelligence for education and its emphasis on assessment and adversity quotient: a review", "Jyoti Mudkanna Gavhane; Reena Pagare"),
    14: ("A comprehensive exploration of artificial intelligence competence frameworks for educators: A critical review", "Tamar Mikeladze; Paulien C. Meijer; Roald P. Verhoeff"),
    15: ("A Systematic Review of Generative AI for Teaching and Learning Practice", "Bayode Ogunleye; Kudirat Ibilola Zakariyyah; Oluwaseun Ajao; Olakunle Olayinka; Hemlata Sharma"),
    16: ("Generative Artificial Intelligence in Tertiary Education: Assessment Redesign Principles and Considerations", "Che Yee Lye; Lyndon Lim"),
    17: ("A Systematic Review of AI-Driven Educational Assessment in STEM Education", "Fan Ouyang; Tuan Anh Dinh; Weiqi Xu"),
    18: ("Transforming Education: A Comprehensive Review of Generative Artificial Intelligence in Educational Settings through Bibliometric and Content Analysis", "Zied Bahroun; Chiraz Anane; Vian Ahmed; Andrew Zacca"),
    19: ("What Is the Impact of ChatGPT on Education? A Rapid Review of the Literature", "Chung Kwan Lo"),
    20: ("Systematic review of research on artificial intelligence applications in higher education – where are the educators?", "Olaf Zawacki-Richter; Victoria I. Marín; Melissa Bond; Franziska Gouverneur"),
}


def numero_articulo(ruta_pdf):
    """Obtiene el identificador numérico del inicio del nombre del PDF."""
    coincidencia = re.match(r"\s*(\d+)\.", ruta_pdf.name)
    if coincidencia is None:
        raise ValueError(f"El PDF no tiene identificador numérico: {ruta_pdf.name}")
    return int(coincidencia.group(1))


def construir_corpus():
    archivos_pdf = sorted(CARPETA_ARTICULOS.glob("*.pdf"), key=numero_articulo)
    if len(archivos_pdf) != TOTAL_ESPERADO:
        raise ValueError(f"Se esperaban {TOTAL_ESPERADO} PDF y se encontraron {len(archivos_pdf)}.")

    registros = []
    for archivo in archivos_pdf:
        identificador = numero_articulo(archivo)
        if identificador not in METADATOS:
            raise ValueError(f"Faltan metadatos para el artículo {identificador}.")

        try:
            texto_pdf = extraer_texto_pdf(str(archivo))
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Falló la lectura del artículo {identificador}: {error}"
            ) from error

        abstract = extraer_abstract(texto_pdf)
        if not abstract or not abstract.strip():
            raise ValueError(
                f"No se pudo extraer el abstract del artículo {identificador}."
            )

        titulo, autores = METADATOS[identificador]
        registros.append({
            "id": identificador,
            "archivo_pdf": archivo.name,
            "titulo": titulo,
            "autores": autores,
            "abstract": abstract.strip(),
            "abstract_normalizado": normalizar_texto(abstract),
        })

    with RUTA_SALIDA.open("w", encoding="utf-8-sig", newline="") as salida:
        campos = ["id", "archivo_pdf", "titulo", "autores", "abstract", "abstract_normalizado"]
        escritor = csv.DictWriter(salida, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)

    return registros


if __name__ == "__main__":
    corpus = construir_corpus()
    print(f"Artículos procesados: {len(corpus)}/{TOTAL_ESPERADO}")
    print(f"Abstracts vacíos: {sum(not fila['abstract'] for fila in corpus)}")
    print(f"Metadatos incompletos: {sum(not fila['titulo'] or not fila['autores'] for fila in corpus)}")
    print(f"CSV generado: {RUTA_SALIDA}")
