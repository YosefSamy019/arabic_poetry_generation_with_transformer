import streamlit as st
import load_model
import time
import re_funs
import numpy as np

STATE_GENERATION = 'gen'
STATE_INPUT = 'input'
STATE_OUTPUT = 'output'
STATE_OUTPUT_TOKENS = 'output_tokens'
STATE_TEMPERATURE = 'temperature'

# تحميل النموذج
@st.cache_resource
def load():
    tokenizer, model = load_model.load_pipeline()
    return tokenizer, model

def main():
    st.session_state[STATE_GENERATION] = st.session_state.get(STATE_GENERATION, False)
    st.session_state[STATE_TEMPERATURE] = st.session_state.get(STATE_TEMPERATURE, 0)
    st.session_state[STATE_OUTPUT] = st.session_state.get(STATE_OUTPUT, "")
    st.session_state[STATE_OUTPUT_TOKENS] = st.session_state.get(STATE_OUTPUT_TOKENS, [])

    st.set_page_config(
        page_title="مولّد الشعر العربي",
        page_icon="🕊️",
        layout="wide",
    )

    # إعداد الاتجاه من اليمين إلى اليسار
    st.markdown(
        """
        <style>
        body, textarea, input, label, div, h1, h2, h3, h4, h5, h6, span, p {
            direction: rtl;
            text-align: right;
            font-family: "Amiri", "Cairo", "Tahoma", sans-serif;
        }
        .stTextArea textarea {
            direction: rtl !important;
            text-align: right !important;
        }
        .stTextInput input {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("مولّد الشعر العربي ✍️")

    st.subheader("هذا التطبيق يُستخدم لتوليد أبيات شعر عربية باستخدام الذكاء الاصطناعي")

    with st.spinner("يتم الآن تحميل النموذج..."):
        load()

    cols = st.columns(2)

    with cols[0]:
        st.session_state[STATE_INPUT] = st.text_area(
            "أدخل بداية البيت الشعري (المطلِع):",
            value=st.session_state.get(STATE_INPUT, ""),
            placeholder="اكتب مثلاً: يا ليلُ، أو يا قلبُ...",
        )

        st.session_state[STATE_TEMPERATURE] = st.radio(
            "هل ترغب في توليد عشوائي (بعض الإبداع في النتيجة)؟",
            options=['لا', 'نعم'],
            index=st.session_state[STATE_TEMPERATURE]
        )
        st.session_state[STATE_TEMPERATURE] = int(st.session_state[STATE_TEMPERATURE] == 'نعم')

        sub_cols = st.columns(2)
        sub_cols[0].button("ابدأ التوليد", type="primary", on_click=generate, disabled=st.session_state[STATE_GENERATION])
        sub_cols[1].button("إلغاء", type="secondary", on_click=cancel, disabled=not st.session_state[STATE_GENERATION])

    with cols[1]:
        st.container(height=10, border=False)
        with st.container(border=True):
            st.text_area(
                "الناتج الشعري:",
                value=st.session_state[STATE_OUTPUT],
                height=300,
            )

    if st.session_state[STATE_GENERATION] is True:
        generate_token()
        # time.sleep(0.05)
        st.rerun()


def generate():
    st.session_state[STATE_OUTPUT_TOKENS] = []
    st.session_state[STATE_GENERATION] = True

def cancel():
    st.session_state[STATE_GENERATION] = False

def generate_token():
    tokenizer, model = load()

    if len(st.session_state[STATE_OUTPUT_TOKENS]) == 0:
        cleaned_input_msg = re_funs.clean_poem_text(st.session_state.get(STATE_INPUT, f'{re_funs.START_TOKEN}'))
        if len(cleaned_input_msg) == 0:
            cleaned_input_msg = re_funs.START_TOKEN + ' ' + re_funs.START_TOKEN
        tokenized_input_msg = cleaned_input_msg.split(' ')

        while len(tokenized_input_msg) < load_model.WINDOW_SIZE:
            tokenized_input_msg = [re_funs.START_TOKEN] + tokenized_input_msg

        st.session_state[STATE_OUTPUT_TOKENS].extend(tokenized_input_msg)

    # الحصول على نافذة الإدخال
    cur_input = st.session_state[STATE_OUTPUT_TOKENS]
    cur_input = cur_input[len(cur_input) - load_model.WINDOW_SIZE:]

    cur_input = re_funs.CLS_TOKEN + ' ' + " ".join(cur_input)
    cur_input_num = tokenizer.texts_to_sequences([cur_input])[0]
    m_prop = model.predict(np.array([cur_input_num]), verbose=0)[0]

    if st.session_state[STATE_TEMPERATURE] == 1:
        ch_num = np.random.choice(len(m_prop), p=m_prop)
    else:
        ch_num = np.argmax(m_prop)

    ch = tokenizer.sequences_to_texts([[ch_num]])[0]
    st.session_state[STATE_OUTPUT_TOKENS].append(ch)
    st.session_state[STATE_OUTPUT] = re_funs.compose_poem_text(" ".join(st.session_state[STATE_OUTPUT_TOKENS]))


if __name__ == "__main__":
    main()
