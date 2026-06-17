import streamlit as st
import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 초기 설정 및 API 연결
# ==========================================
st.set_page_config(page_title="Pro AI Music Curator", layout="wide", page_icon="🎧")

# API 클라이언트 초기화 (secrets.toml 파일 필요)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]
))

# 추천에 사용할 11가지 오디오 특징 칼럼 목록
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]


# ==========================================
# 2. 데이터 로드 및 정규화 (실제 내 파일 적용 버전)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    """
    내 컴퓨터에 있는 music_database.csv 파일을 읽어와 정규화합니다.
    """
    # 💡 이 줄을 통해 실제 질문자님의 파일을 읽어옵니다!
    df_raw = pd.read_csv("spotify_songs2.csv", encoding="cp949")
    
    # 데이터 정규화 진행
    scaler = MinMaxScaler()
    df_normalized = df_raw.copy()
    df_normalized[FEATURES] = scaler.fit_transform(df_raw[FEATURES])
    
    return df_raw, df_normalized, scaler
    
    # 화면 출력용 원본 df, 계산용 정규화 df, 스케일러 객체 반환
    return df_raw, df_normalized, scaler


# ==========================================
# 3. 핵심 함수: 텍스트 -> 벡터 변환 (ChatGPT)
# ==========================================
def get_features_from_prompt(user_prompt):
    system_instruction = """
    당신은 음악 오디오 데이터 분석 전문가입니다. 
    사용자의 텍스트를 분석하여 Spotify의 11가지 오디오 특징을 0.0에서 1.0 사이의 수치로 추정하세요.
    반드시 아래의 JSON 형식으로만 응답해야 하며 다른 텍스트는 출력하지 마세요.
    {"danceability": 0.5, "energy": 0.5, "key": 0.5, "loudness": 0.5, "mode": 0.5, "speechiness": 0.5, "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.5, "valence": 0.5, "tempo": 0.5}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # 또는 gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        response_format={ "type": "json_object" }
    )
    
    return json.loads(response.choices[0].message.content)


# ==========================================
# 4. 핵심 함수: 참고 노래 -> 벡터 변환 및 정규화
# ==========================================
def get_scaled_reference_features(song_name, scaler):
    # 1. Spotify API에서 곡 검색
    results = sp.search(q=song_name, limit=1, type='track')
    if not results['tracks']['items']:
        return None
    
    track_id = results['tracks']['items'][0]['id']
    track_info = results['tracks']['items'][0]
    
    # 2. 오디오 특징 가져오기
    features_raw = sp.audio_features(track_id)[0]
    
    # 3. 필요한 11개 특징만 딕셔너리로 추출
    ref_dict = {f: features_raw[f] for f in FEATURES}
    
    # 4. 데이터베이스와 동일한 스케일로 정규화(Transform)
    df_ref = pd.DataFrame([ref_dict])
    df_ref[FEATURES] = scaler.transform(df_ref[FEATURES])
    
    return df_ref.iloc[0].to_dict(), track_info['name'], track_info['artists'][0]['name']


# ==========================================
# 5. Streamlit 메인 UI 및 로직
# ==========================================
st.title("🎧 Pro AI Music Curator")
st.markdown("당신의 기분과 참고하고 싶은 노래를 바탕으로 완벽한 곡을 추천해 드립니다.")

# 데이터 및 스케일러 로드
df_raw, df_normalized, scaler = load_and_preprocess_data()

# UI 레이아웃 구성
col1, col2 = st.columns(2)
with col1:
    user_prompt = st.text_area("어떤 느낌의 곡을 원하시나요?", placeholder="비오는 날 카페에서 들을 법한 차분하고 어쿠스틱한 노래를 원해.", height=100)
    
    # 프로 모델을 위한 고급 설정 (가중치 조절 슬라이더)
    blend_ratio = st.slider("참고 노래 반영 비율 (%)", min_value=0, max_value=100, value=50, step=10)

with col2:
    ref_song = st.text_input("참고할 노래가 있다면 적어주세요 (선택)", placeholder="곡 제목 및 아티스트 (예: Ditto NewJeans)")

# 실행 버튼
if st.button("추천 시작", type="primary", use_container_width=True):
    if not user_prompt:
        st.warning("원하시는 느낌을 프롬프트에 먼저 적어주세요!")
    else:
        with st.spinner("AI가 음악의 파동을 분석하고 있습니다..."):
            
            # 1. 텍스트 프롬프트를 벡터로 변환
            prompt_features_dict = get_features_from_prompt(user_prompt)
            target_vector = [prompt_features_dict[f] for f in FEATURES]
            
            # 2. 참고 노래 처리 및 벡터 혼합
            if ref_song:
                ref_result = get_scaled_reference_features(ref_song, scaler)
                
                if ref_result:
                    ref_features_dict, found_name, found_artist = ref_result
                    st.success(f"🎵 참고 노래 감지 성공: **{found_name}** by {found_artist}")
                    
                    # 사용자가 설정한 슬라이더 비율에 따라 가중 평균 계산
                    prompt_weight = (100 - blend_ratio) / 100.0
                    ref_weight = blend_ratio / 100.0
                    
                    target_vector = [
                        (prompt_features_dict[f] * prompt_weight) + (ref_features_dict[f] * ref_weight) 
                        for f in FEATURES
                    ]
                else:
                    st.error("Spotify에서 해당 노래를 찾을 수 없어 텍스트 프롬프트만으로 추천을 진행합니다.")

            # 3. 코사인 유사도 계산 (정규화된 DB와 타겟 벡터 비교)
            database_vectors = df_normalized[FEATURES].values
            similarities = cosine_similarity([target_vector], database_vectors)
            
            # 4. 결과 출력
            # 계산은 정규화된 데이터로 했지만, 보여주는 것은 원본 데이터(df_raw)에 붙여서 보여줍니다.
            df_raw['similarity'] = similarities[0]
            top_recommendations = df_raw.sort_values(by='similarity', ascending=False).head(5)
            
            st.divider()
            st.subheader("✨ 당신을 위한 맞춤 추천 결과")
            
            # 출력할 칼럼 정리 (인덱스 숨기고 깔끔하게)
            display_df = top_recommendations[['name', 'artist', 'similarity']].copy()
            display_df.columns = ['곡 제목', '아티스트', '추천 일치율']
            
            st.dataframe(
                display_df.style.format({'추천 일치율': '{:.2%}'}),
                use_container_width=True,
                hide_index=True
            )
