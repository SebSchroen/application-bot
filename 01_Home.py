import streamlit as st
import yaml
from modules.InfoLoader import InfoLoader
from modules.VectorDB import VectorDB
import logging

@st.cache_resource
def configure_logging(file_path=None, streaming=None, level=logging.INFO):
    '''
    Initiates the logger, runs once due to caching
    '''
    # streamlit_root_logger = logging.getLogger(st.__name__)

    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not len(logger.handlers):
        # Add a filehandler to output to a file
        if file_path:
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('Added filer_handler')

        # Add a streamhandler to output to console
        if streaming:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.info('Added stream_handler')
    

    return logger

def initialize_session_state():
    '''
    Handles initializing of session_state variables
    '''
    # Load config if not yet loaded
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)
            
    # Create an API call counter, to cap usage
    if 'usage_counter' not in st.session_state:
        st.session_state.usage_counter = 0
    st.session_state.openai_api_key_host = st.secrets["apikey"]

        
@st.cache_resource
def get_resources():
    '''
    Initializes the customer modules
    '''
    return InfoLoader(st.session_state.config), VectorDB(st.session_state.config)

def main():
    '''
    Main Function for streamlit interface
    '''
    # Load configs, logger, classes
    st.set_page_config(page_title="Bewerbungsbot ")
    initialize_session_state()    
    if st.session_state.config['local']:
        logger = configure_logging('app.log')
    else: 
        logger = configure_logging(streaming=True)
    loader, vector_db = get_resources()  

    #------------------------------------ SIDEBAR ----------------------------------------#
    with st.sidebar:
        # API option, whether to use host's API key (must be enabled by config), and also to cap usage
        openai_api_key = st.session_state.openai_api_key_host
        password = st.text_input(label = "Passwort", help = "Passwort hier eingeben.")
        # Document uploader
        uploaded_files = st.file_uploader(
            label = 'Dokumente hier hochladen', 
            help = 'Bereits existierende Dokumente werden überschrieben',
            type = ['pdf', 'txt', 'docx', 'srt'], 
            accept_multiple_files=True
            )

        if st.button('Hochladen', type='primary') and (uploaded_files) and password == st.secrets['password']:
            with st.status('Lade hoch (das kann einen Moment dauern)...)', expanded=True) as status:
                try:
                    st.write("Dokumente werden verarbeitet...")
                    loader.get_chunks(uploaded_files)
                    st.write("Embeddings werden erstellt...")
                    vector_db.create_embedding_function(openai_api_key)
                    vector_db.initialize_database(loader.document_chunks_full, loader.document_names)

                except Exception as e:
                    logger.error('Exception during Splitting / embedding', exc_info=True)
                    status.update(label='Error occured.', state='error', expanded=False)
                else:
                    # If successful, increment the usage based on number of documents
                    if openai_api_key == st.session_state.openai_api_key_host:
                        st.session_state.usage_counter += len(loader.document_names)
                        logger.info(f'Current usage counter: {st.session_state.usage_counter}')
                    logger.info(f'Erfolgreich hochgeladen: {loader.document_names}')
                    status.update(label='Embedding vollständig!', state='complete', expanded=False)
        temperature = st.select_slider(
        'Kreativität', 
        options=[x / 10 for x in range(0, 21)],
        value= 1.0,
        help='Je höher der Wert, desto größer ist der Zufallseinfluss beim Erstellen des Anschreibens \n\
            Ein Wert von 1 ist ideal.',
        # disabled = missing_api_key
        )

    #------------------------------------- MAIN PAGE -----------------------------------------#
    st.markdown("## :rocket: Willkommen zu deinem Bewerbungs-Assistenten")

    # Info bar
    if vector_db.document_names:
        doc_name_display = ''
        for doc_count, doc_name in enumerate(vector_db.document_names):
            doc_name_display += str(doc_count+1) + '. ' + doc_name + '\n\n'
    else:
        doc_name_display = 'Noch keine Dokumente hochgeladen!'
    st.info(f"Hochgeladene Dokumente: \n\n {doc_name_display}", icon='ℹ️')
    if (not openai_api_key.startswith('sk-')) or (openai_api_key=='NA'):
        st.write('Enter your API key on the sidebar to begin')

    # Query form and response
    with st.form('my_form'):
        user_instructions = st.text_area('Hier die Anweisungen eingeben  und mit einer Erfahrung ergänzen:', value = 'Du bist ein Bewerbungs-Assistent und erstellst auf Basis der Stellenausschreibung ein Anschreiben.')
        user_information = st.text_area('Schreib etwas über dich:', value = '- Ich habe 20 Jahre Berufserfahrung  \n - Ich heiße Max Mustermann \n - Der Ansprechpartner für die Stelle ist Maxim Mustermensch')
        user_input = st.text_area('Prompt:', value='Erstelle mir ein Anschreiben.')
        source = 'Uploaded documents'

        # Select for model and prompt template settings
        # prompt_mode = st.selectbox(
        #     'Choose mode of prompt', 
        #     options = ('Restricted', 'Unrestricted'),
        #     help='Restricted mode will reduce chances of LLM answering using out of context knowledge',
        #     # disabled = missing_api_key
        #     )
        prompt_mode = 'Unrestricted'

        
        # Display error if no API key given
        if not openai_api_key.startswith('sk-'):
            if openai_api_key == 'NA':
                st.warning('Host key currently not available, please use your own OpenAI API key!', icon='⚠')
            else:
                st.warning('Please enter your OpenAI API key!', icon='⚠')

        #----------------------------------------- Submit a prompt ----------------------------------#
        if st.form_submit_button('Absenden', type='primary') and openai_api_key.startswith('sk-'):
            with st.spinner('Lade...'):
                try:
                    result = None
                    vector_db.create_llm(
                        openai_api_key,
                        temperature
                    )
                    vector_db.create_chain(
                        prompt_mode,
                        source
                    )
                    result = vector_db.get_response(user_instructions + " " + user_information + " " +  user_input)
                except Exception as e:
                    logger.error('Exception during Querying', exc_info=True)
                    st.error('Error occured, unable to process response!', icon="🚨")

            if result:
                # Display the result
                st.info('Anschreiben:', icon='📕')
                st.info(result["result"])
                #st.write(' ')
                #st.info('Sources', icon='📚')
                # for document in result['source_documents']:
                #     st.write(document.page_content + '\n\n' + document.metadata['source'] + ' (pg ' + str(document.metadata.get('page', 'na')) + ')')
                #     st.write('-----------------------------------')

if __name__ == '__main__':
   main()
