from fpdf import FPDF

def generate_pdf_report(memory, output_path="/content/project/Agentes_Autonomos_Relatorio.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,10, "Relatório - Agente EDA", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    interactions = memory.get_all(limit=6)
    if not interactions:
        pdf.multi_cell(0,6, "Sem interações.")
    for ts, q, a, meta in interactions:
        pdf.multi_cell(0,6, f"Pergunta: {q}")
        pdf.multi_cell(0,6, f"Resposta (resumo): {a}")
        pdf.ln(2)
   import os
    # garante que salva no diretório atual da aplicação
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(os.path.basename(output_path))
    
    return output_path
