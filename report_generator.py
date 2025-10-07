from fpdf import FPDF
import os

def generate_pdf_report(memory, output_path="Relatorio_Agente_IA.pdf"):
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatório - Agente de Análise de Dados", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    data = memory.get_all(limit=10)
    if not data:
        pdf.multi_cell(0, 8, "Nenhuma interação registrada.")
    else:
        for ts, q, a, meta in data:
            pdf.multi_cell(0, 8, f"Pergunta: {q}")
            pdf.multi_cell(0, 8, f"Resposta: {a}")
            pdf.ln(5)

    pdf.output(output_path)
    return output_path
