from fpdf import FPDF
import os

def generate_pdf_report(memory, output_path="Agentes_Autonomos_Relatorio.pdf"):
    # Garante que o diretório de saída existe
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Relatório - Agente EDA", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, "Este relatório apresenta um resumo das análises realizadas pelo agente de Exploração de Dados (EDA).")
    pdf.ln(6)

    # Interações registradas na memória
    interactions = memory.get_all(limit=10)
    if not interactions:
        pdf.multi_cell(0, 6, "Nenhuma interação foi registrada ainda.")
    else:
        for ts, q, a, meta in interactions:
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, f"Pergunta: {q}")
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 6, f"Resposta resumida: {a}")
            pdf.ln(4)

    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Conclusão", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, "As análises apresentadas indicam padrões, distribuições, correlações e possíveis outliers detectados pelo agente.")

    # Salva o PDF no diretório atual
    pdf.output(output_path)
    return output_path
