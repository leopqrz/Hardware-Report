"""
Contains all functions related to graph plotting and PDF file generation.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from matplotlib import pyplot as plt
import seaborn as sns
import requests
import datetime as dt

class GraphBuilder:
    """
    Graphic related logic
    """

    def lineplot(self, device, data, filename):
        """
        Line plots for CPU and memory data
        """
        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(12, 8)
        fig.suptitle(f'{device} Usage vs Time', fontname="Times New Roman",
                     fontweight="bold", fontsize=16)

        sns.lineplot(data=data.iloc[:, :-1], ax=axes[0])
        sns.lineplot(data=data.iloc[:, -1],
                     ax=axes[1], label=f"Total {device}")

        axes[0].set_title(f'{device} Usage')
        axes[1].set_title(f'Total {device} Usage')

        def fill_plotted_area(axis=None, alpha=.2, **kwargs):
            """
            Displays the lineplot with its filled area
            """
            if axis is None:
                axis = plt.gca()
            for line in axis.lines:
                x_axis, y_axis = line.get_xydata().T
                axis.fill_between(x_axis, 0, y_axis, color=line.get_color(),
                                  alpha=alpha, **kwargs)

        fill_plotted_area(axis=axes[1], alpha=.2)

        if device == 'CPU':
            for axis in axes.flat:
                axis.set(xlabel='Datetime',
                         ylabel=f'{device} (%)', ylim=(0, 100))
                axis.tick_params(axis='x', labelrotation=45)
        elif device in ('Memory', 'Swap'):
            axes[0].set(xlabel='Datetime', ylabel=f'{device} (Gb)')
            axes[0].tick_params(axis='x', labelrotation=45)
            axes[1].set(xlabel='Datetime',
                        ylabel=f'{device} (%)', ylim=(0, 100))
            axes[1].tick_params(axis='x', labelrotation=45)

        fig.tight_layout()
        plt.close(fig)
        fig.savefig(filename)


class PdfBuilder:
    """
    PDF related logics
    """
    pagesize = A4
    # Gets the logo image location on the web
    logo_url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
    # Creates the Image object for the logo, with its size setup
    logo = Image(requests.get(logo_url, stream=True).raw, 2*inch, 0.5*inch)
    logo.hAlign = 'LEFT'  # Set the logo on the left side of the PDF

    def create_template(self, current_formatted_time, pagesize):
        """
        Creates the PDF template.
        :args:
            PDF filename: <str>
            pagesize: <object> Page format (A4, LETTER)
            rightMargin: <float> Right margin in inches
            leftMargin: <float> Left margin in inches
            topMargin: <float> Top margin in inches
            bottomMargin: <float> Bottom margin in inches
        """
        return SimpleDocTemplate(f"Hardware_Report_{current_formatted_time}.pdf",
                                 pagesize=pagesize,
                                 rightMargin=0.5*inch,
                                 leftMargin=0.5*inch,
                                 topMargin=0.5*inch,
                                 bottomMargin=0.5*inch)

    def format_text(self, text):
        """
        Format a text for the PDF file and replace
        string commands to html formatting.
        e.g:
            "\n" to "<br />"
            "\t" to "&nbsp;"*4
        :args:
            text: <str> Text to be written on the PDF
        :output:
            <object> Object with the formatted text for the PDF
        """
        return Paragraph(text.replace("\n", "<br />").replace("\t", "&nbsp;"*4),
                         getSampleStyleSheet()['Normal'])

    def format_image(self,
                     filename,
                     position="CENTER",
                     width=A4[0]-inch,
                     height=(A4[0]-inch)*(8/12)):
        """
        Format an image for the PDF file.
        :args:
            filename: <str> Image filename
            position: <str> Image position (CENTER, LEFT, RIGHT)
            width: <float> Image width in inches
            height: <float> Image height in inches
        :output:
            <object> Object with the formatted image for the PDF
        """
        plot = Image(filename, width, height)
        plot.hAlign = position
        return plot

    def format_table(self, data):
        """
        Format a table for the PDF file.
        :args:
            data: <list(str)> List of strings to be on the table of the PDF file
        :output:
            <object> Object with the formatted table for the PDF
        """
        formatted_table = Table(data, style=[
            # Creates a black grid for the whole table
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Align the whole text to be centered
            ('ALIGN', (0, 0), (-1, -1), 'CENTRE'),
            # Draw a gray background for the first row
            ('BACKGROUND', (0, 0), (-1, 0), colors.gray)
        ])
        return formatted_table

    def unformatted_table(self, data):
        """
        Creates an unformatted table (No grids, no background).
        :args:
            data: <list(str)> List of strings to be on the table of the PDF file
        :output:
            <object> Object with the unformatted table for the PDF

        """
        return Table(data)

    def go_next_page(self):
        """
        Put the next part of the PDF (text, table or image) on the next page
        """
        return PageBreak()

    def build_pdf(self, canvas, parts):
        """
        Builds the PDF.
        :args:
            canvas: <object> Object canvas with the PDF template
            parts: <list(object)> List of objects (str, Image, Table) that are part of the PDF
        :output:
            <object> Object with the final PDF
        """
        canvas.build(parts)



# General functions

def addMins(tm, mins):
    fulldate = dt.datetime(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + dt.timedelta(minutes=mins)
    return fulldate.time()