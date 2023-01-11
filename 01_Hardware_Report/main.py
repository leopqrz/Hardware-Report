"""Main script for the FronEnd using Streamlit."""
import datetime as dt

import fitz
import matplotlib.pyplot as plt
import streamlit as st
import utils
from PIL import Image
from report import HardwareReport

img = 'https://i.gifer.com/74pZ.gif'

st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title='Intro',
    page_icon=img,
)
with st.spinner('Wait for it...'):

    # --------------------------- Layout setting ---------------------------
    # Create an empty container in the sidebar
    window_selection_c = st.sidebar.container()
    # Add the gif image to the sidebar container
    window_selection_c.image(img, caption='', width=300)
    # Add a title to the sidebar container
    window_selection_c.markdown('# Computer Hardware Report')
    # Add a subtitle to the sidebar container
    window_selection_c.markdown('### Select the time range to be analyzed:')
    # Split the container into two columns for start and end date
    sub_columns = window_selection_c.columns(2)

    # --------------------------- Time window selection ----------------------
    START_DATE = dt.datetime.now().date()
    STOP_DATE = START_DATE
    START_DATE = sub_columns[0].date_input(
        'From', value=START_DATE, max_value=START_DATE
    )
    STOP_DATE = sub_columns[1].date_input(
        'To', value=STOP_DATE, min_value=START_DATE)

    START_TIME = dt.datetime.now().time()
    STOP_TIME = utils.addMins(START_TIME, 1)
    START_TIME = sub_columns[0].time_input(
        'From', value=START_TIME, label_visibility='collapsed'
    )
    STOP_TIME = sub_columns[1].time_input(
        'To', value=STOP_TIME, label_visibility='collapsed'
    )

    START = dt.datetime.combine(START_DATE, START_TIME)
    STOP = dt.datetime.combine(STOP_DATE, STOP_TIME)

    create_report = sub_columns[0].button('Create Report')

    def displayPDF(filename: str) -> None:
        """Convert the report PDF file to images and displays to the frontend.

        :param filename: Path with the filename of the PDF file
        """
        doc = fitz.open(filename)
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)

        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            # set the mode depending on alpha
            mode = 'RGBA' if pix.alpha else 'RGB'
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            st.image(img, caption='')
        doc.close()

    if create_report:
        report = HardwareReport(
            start_time=START,
            stop_time=STOP,
            system=True,
            cpu=True,
            memory=True,
            disk=True,
            network=True,
        )
        report.generate_pdf()
        displayPDF(report.canvas.filename)
        st.success('Done!')
