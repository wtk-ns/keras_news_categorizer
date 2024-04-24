from model import Categorizer
import PySimpleGUI as sg


def predicate_text(text):
    prepared_text = categorizer.prepare_text_ro_predict(text)
    predicted = model.predict(prepared_text)[0]
    category_probabilities = [f"{cat} ({prob:.2%})" for cat, prob in zip(category_list, predicted)]
    formatted_output = ", ".join(category_probabilities)
    return formatted_output


def create_ui():
    layout = [[sg.Text('News category predicator:')],
              [sg.Text('Input title:'), sg.InputText(enable_events=True, key='input')],
              [sg.Text('', key='predict')],
              [sg.Button('Close')]]
    window = sg.Window('News category', layout)

    last_len = 0

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Close':
            break
        if event == 'input':
            text = values['input']
            if last_len > len(text):
                window['predict'].update('')
                last_len = len(text)
                continue

            last_len = len(text)

            if len(text) < 3 or text[-1] != ' ':
                continue
            window['predict'].update(predicate_text(text))

    window.close()


if __name__ == '__main__':
    DATASET_NAME, DELIM = 'bbc-news-data.csv', '\t'
    MODEL_PATH = 'model.pkl'

    categorizer = Categorizer()
    # categorizer.train_model(DATASET_NAME, DELIM, MODEL_PATH)
    model, category_list = categorizer.load_trained_model(DATASET_NAME, DELIM, MODEL_PATH)
    create_ui()
