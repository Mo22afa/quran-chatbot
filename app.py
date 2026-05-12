import streamlit as st

from matcher import QuranMatcher


AUDIO_CDN_TEMPLATE = "https://cdn.islamic.network/quran/audio/128/{edition}/{ayah_no_quran}.mp3"
DEFAULT_AUDIO_EDITION = "ar.alafasy"


def build_audio_url(result, edition=DEFAULT_AUDIO_EDITION):
    ayah_no_quran = result.get("ayah_no_quran")
    if not ayah_no_quran:
        return None
    return AUDIO_CDN_TEMPLATE.format(
        edition=edition,
        ayah_no_quran=int(ayah_no_quran),
    )


def render_message(message):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("audio_url"):
            st.audio(message["audio_url"], format="audio/mp3")


st.set_page_config(page_title="Quran Ayah Correction Chatbot", page_icon="📖")

st.title("Quran Ayah Correction Chatbot")


@st.cache_resource(show_spinner="Loading Quran matcher...")
def load_matcher():
    return QuranMatcher("data/quran.csv")


matcher = load_matcher()

st.subheader("بحث سريع بالحروف الأولى")
prefix_input = st.text_input(
    "اكتب أول حروف من الآية",
    placeholder="مثال: الحمد، قل هو، صم",
    key="prefix_input",
)

if prefix_input:
    prefix_results = matcher.prefix_search(prefix_input, top_k=8)

    if not prefix_results:
        st.info("لا توجد نتائج مطابقة للحروف المدخلة.")
    else:
        for index, item in enumerate(prefix_results, start=1):
            st.markdown(
                f"**{index}. {item['surah_name']} - آية {item['ayah_number']}**  \n"
                f"{item['text']}"
            )

        first_audio_url = build_audio_url(prefix_results[0])
        if first_audio_url:
            st.audio(first_audio_url, format="audio/mp3")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "اكتب آية أو جزء من آية، وسأرجع لك أقرب نص صحيح من قاعدة البيانات مع التلاوة الصوتية.",
        }
    ]

for message in st.session_state.messages:
    render_message(message)

user_input = st.chat_input("اكتب الآية هنا...")

if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    prefix_results = matcher.prefix_search(user_input, top_k=5)
    if prefix_results and prefix_results[0].get("final_score", 0) >= 70:
        results = prefix_results
    else:
        results = matcher.search(user_input, top_k=5)

    audio_url = None
    if not results:
        reply = "اكتب نصا عربيا واضحا من الآية أو جزءا منها."
    else:
        best = results[0]
        audio_url = build_audio_url(best)
        reply = f"""الصحيح:

{best["text"]}

السورة: {best["surah_name"]}
رقم الآية: {best["ayah_number"]}
نسبة التشابه التقريبية: {best["final_score"]:.1f}%
"""

        if len(results) > 1:
            alternatives = []
            for item in results[1:4]:
                alternatives.append(
                    f'- {item["surah_name"]}، آية {item["ayah_number"]}: {item["text"]}'
                )
            reply += "\nاقتراحات قريبة:\n" + "\n".join(alternatives)

    assistant_message = {
        "role": "assistant",
        "content": reply,
        "audio_url": audio_url,
    }
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)
