import PyPDF2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# Open the input PDF file
input_file_name = 'input.pdf'
pdf_file = open(input_file_name, 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Create a new PDF file for output
output_file_name = 'output.pdf'
pdf_canvas = SimpleDocTemplate(output_file_name, pagesize=letter)
elements = []

# Loop through the pages in the input PDF file
for page_number in range(pdf_reader.numPages):
    page = pdf_reader.getPage(page_number)
    page_text = page.extractText()

    # Tokenize the text and perform name entity recognition
    tokens = word_tokenize(page_text)
    tagged = pos_tag(tokens)
    ne_tagged = ne_chunk(tagged)

    # Create a new paragraph for each line of text with the name entity label marked under it
    paragraph_styles = getSampleStyleSheet()
    normal_style = paragraph_styles['Normal']
    ne_label_style = paragraph_styles['Code']
    elements.append(normal_style('Page %d' % (page_number+1)))
    for i in range(len(tokens)):
        token = tokens[i]
        ne_label = ''
        if isinstance(ne_tagged[i], Tree):
            ne_label = ne_tagged[i].label()
        line = normal_style(token + '\n')
        ne_label_line = ne_label_style(ne_label + '\n')
        elements.append(line)
        elements.append(ne_label_line)

    # Add a spacer between pages
    if page_number < pdf_reader.numPages-1:
        elements.append(Spacer(1, 0.2 * inch))

# Write the output PDF file
pdf_canvas.build(elements)
pdf_file.close()
